package dev.dream.companion.data

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

data class SessionState(
    val link: Link = Link.IDLE,
    val error: String = "",
    val snapshot: Snapshot? = null,
    val running: String? = null,
    /** Oldest first. */
    val tasks: List<Task> = emptyList(),
    val feed: Feed = Feed(),
    val stopping: Set<String> = emptySet(),
    val notice: String = "",
    val choice: ModelChoice? = null,
    val hasToken: Boolean = false,
    val endpoint: String = CredentialStore.DEFAULT_ENDPOINT,
    /** Phone time at which each task was first seen finished; tasks already finished at connect time are absent. */
    val endedAt: Map<String, Long> = emptyMap(),
) {
    val choices get() = snapshot?.choices ?: emptyList()
    /** The running task, else the next one the scheduler would pick (priority DESC, created ASC), else the latest. */
    val shown: Task?
        get() = tasks.find { it.id == running }
            ?: tasks.filter { it.runnable }.sortedWith(compareByDescending<Task> { it.priority }.thenBy { it.created }).firstOrNull()
            ?: tasks.lastOrNull()
    val everConnected get() = snapshot != null
}

/** Events the app turns into notifications or spoken replies; produced only from real backend state changes. */
sealed interface SessionEvent {
    data class TaskEnded(val task: Task, val state: String, val reason: String?) : SessionEvent
    data class SafetyStop(val reason: String) : SessionEvent
    data class ConnectionChanged(val online: Boolean) : SessionEvent
    data class ToolProblem(val title: String, val body: String?) : SessionEvent
}

/**
 * Polls the backend exactly like the web UI (status, tasks, snapshot, AG-UI events) while the app is visible.
 * In the background it keeps a slower poll only while a task is runnable, so notifications reach the phone, and stops
 * entirely otherwise: no wakelock, no foreground service. The robot keeps executing on the backend regardless.
 */
class Session(private val store: CredentialStore, private val scope: CoroutineScope, private val onEvent: (SessionEvent) -> Unit) {
    private val _state = MutableStateFlow(SessionState(choice = store.choice, hasToken = store.token != null, endpoint = store.endpoint))
    val state: StateFlow<SessionState> = _state

    private var api: DreamApi? = store.token?.let { DreamApi(store.endpoint, it) }
    private var cursor = 0
    private var streamId: String? = null
    private var loop: Job? = null
    private val gate = Mutex()
    @Volatile private var foreground = true
    private var backgroundSince = 0L

    fun currentApi() = api

    fun start() {
        if (loop?.isActive == true || api == null) return
        loop = scope.launch {
            while (true) {
                poll()
                val s = _state.value
                if (s.link == Link.UNAUTHORISED) break
                val active = s.tasks.any { it.runnable }
                if (!foreground) {
                    // Background: follow an active task for at most 30 minutes, then stop until the app is reopened.
                    if (!active || System.currentTimeMillis() - backgroundSince > 30 * 60_000) break
                    delay(5_000)
                } else delay(if (s.link == Link.ONLINE) 1_200 else 3_500)
            }
        }
    }

    fun setForeground(value: Boolean) {
        foreground = value
        if (!value) backgroundSince = System.currentTimeMillis()
        if (value) start()
    }

    fun connect(endpoint: String, token: String) {
        store.endpoint = endpoint
        store.token = token
        loop?.cancel(); loop = null
        api = DreamApi(endpoint, token)
        cursor = 0; streamId = null
        _state.update { SessionState(link = Link.CONNECTING, choice = store.choice, hasToken = true, endpoint = endpoint) }
        start()
    }

    fun forget() {
        loop?.cancel(); loop = null; api = null; store.token = null
        _state.update { SessionState(choice = store.choice, endpoint = store.endpoint) }
    }

    fun select(choice: ModelChoice) { store.choice = choice; _state.update { it.copy(choice = choice) } }

    /** The stored choice while it is still valid, else the default provider, else the first available one. */
    fun effectiveChoice(s: SessionState = _state.value): ModelChoice? {
        val choices = s.choices
        s.choice?.let { c -> choices.find { it.id == c.provider && it.available }?.let { e -> if ((e.models.isEmpty() && c.model == null) || c.model in e.models) return c } }
        val ids = listOfNotNull(s.snapshot?.provider) + choices.map { it.id }
        for (id in ids) choices.find { it.id == id && it.available }?.let { return ModelChoice(it.id, it.defaultChoice) }
        return null
    }

    private suspend fun poll() = gate.withLock {
        val a = api ?: return@withLock
        val before = _state.value
        // Keep showing the last feed while offline; it is rebuilt from the start of the event stream once the backend answers.
        if (before.link != Link.ONLINE) { cursor = 0; _state.update { it.copy(link = if (it.everConnected) it.link else Link.CONNECTING) } }
        try {
            val status = a.status(); val tasks = a.tasks(); val snap = a.snapshot()
            var ev = a.events(cursor)
            var restarted = false
            if (streamId != null && streamId != ev.optString("stream_id")) { cursor = 0; restarted = true; ev = a.events(0) }
            streamId = ev.optString("stream_id")
            val initial = cursor == 0
            cursor = ev.optInt("cursor", cursor)
            val events = ev.optJSONArray("events").objects()
            var feed = if (before.link != Link.ONLINE || restarted) Feed() else _state.value.feed
            var notices = emptyList<Notice>()
            if (events.isNotEmpty()) { val r = applyEvents(feed, events); feed = r.first; notices = r.second }
            val running = status.optStringOrNull("running")
            val ordered = tasks.sortedBy { it.created }
            val now = System.currentTimeMillis()
            _state.update { s ->
                val ended = s.endedAt.toMutableMap()
                if (s.everConnected) ordered.filter { !it.runnable && it.id !in ended && s.tasks.find { p -> p.id == it.id }?.runnable != false }.forEach { ended[it.id] = now }
                s.copy(link = Link.ONLINE, error = "", snapshot = snap, running = running, tasks = ordered, feed = feed, endedAt = ended,
                    stopping = s.stopping.filter { id -> ordered.find { it.id == id }?.state in setOf("running", "queued") }.toSet())
            }
            if (!initial) {
                for (n in notices) {
                    val t = ordered.find { it.id == n.thread }
                    if (n.title.startsWith("Runtime reports") || n.title.startsWith("Task ")) {
                        if (t != null && n.tone != "info") onEvent(SessionEvent.TaskEnded(t, feed.threads[t.id]?.endState ?: t.state, n.body))
                    } else if (n.tone == "warn" || n.tone == "bad") onEvent(SessionEvent.ToolProblem(n.title, n.body))
                }
            }
            if (before.snapshot?.stopReason == null && snap.stopReason != null && before.everConnected) onEvent(SessionEvent.SafetyStop(snap.stopReason))
            if (before.link == Link.OFFLINE) onEvent(SessionEvent.ConnectionChanged(true))
        } catch (e: ApiError) {
            val unauthorised = e.status == 401
            _state.update { it.copy(link = if (unauthorised) Link.UNAUTHORISED else Link.OFFLINE, error = if (unauthorised) "The access token was rejected. Check the token and try again." else e.message ?: "Connection failed.") }
            if (before.link == Link.ONLINE && !unauthorised) onEvent(SessionEvent.ConnectionChanged(false))
        }
    }

    suspend fun submit(text: String): Boolean {
        val a = api ?: return false
        val choice = effectiveChoice() ?: run { _state.update { it.copy(notice = "No provider is available for new tasks. Check the backend configuration.") }; return false }
        return try {
            a.submit(text, choice); _state.update { it.copy(notice = "") }; poll(); true
        } catch (e: ApiError) { _state.update { it.copy(notice = e.message ?: "The task could not be submitted.") }; false }
    }

    /** Pause / resume / cancel; they take effect at the robot's next safe boundary (the backend scheduler decides). */
    suspend fun act(id: String, action: String): String? {
        val a = api ?: return "Not connected"
        return try {
            val r = a.act(id, action)
            if (r.optString("result") in setOf("pausing", "cancelling")) _state.update { it.copy(stopping = it.stopping + id) }
            poll(); null
        } catch (e: ApiError) { "Could not $action the task: ${e.message}" }
    }

    fun clearNotice() = _state.update { it.copy(notice = "") }
}
