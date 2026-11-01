package dev.dream.companion.data

/** DREAM's visible state, derived only from the task, the agent-event feed, the safety stop and the link (as on the web). */
enum class PresenceState { OFFLINE, IDLE, LISTENING, THINKING, NAVIGATING, OBSERVING, INTERACTING, VERIFYING, RECOVERING, PAUSED, WAITING, COMPLETED, FAILED, SAFETY_STOPPED }

data class Presence(val state: PresenceState, val title: String, val detail: String, val tool: String? = null)

private const val RECENT_S = 90

/** `endedAtMs` is the phone time at which this app first saw the task finish (null if it ended before the app connected). */
fun presenceFor(task: Task?, thread: Thread?, safetyStop: String?, link: Link, listening: Boolean, endedAtMs: Long? = null): Presence {
    if (link != Link.ONLINE) return Presence(PresenceState.OFFLINE, if (link == Link.CONNECTING) "Connecting" else "Not connected", if (link == Link.CONNECTING) "Reaching the robot backend" else "No live session")
    if (listening) return Presence(PresenceState.LISTENING, "Listening", "Speak a goal; it appears in the composer")
    if (safetyStop != null) return Presence(PresenceState.SAFETY_STOPPED, "Safety stopped", safetyStop)
    if (task == null) return Presence(PresenceState.IDLE, "Ready", "Waiting for a goal")
    val recent = endedAtMs != null && System.currentTimeMillis() - endedAtMs < RECENT_S * 1000
    val last = mapOf("succeeded" to "the runtime reported success", "failed" to "it failed", "cancelled" to "it was cancelled")[task.state]
    if (last != null && !recent) return Presence(PresenceState.IDLE, "Ready", "Last task “${task.text}”: $last")
    return when (task.state) {
        "succeeded" -> Presence(PresenceState.COMPLETED, "Task ended", "The runtime reports success")
        "failed" -> Presence(PresenceState.FAILED, "Task failed", task.result ?: "See the mission for details")
        "cancelled" -> Presence(PresenceState.IDLE, "Ready", "The last task was cancelled")
        "queued" -> Presence(PresenceState.WAITING, "Queued", "Waiting for the robot to be free")
        "suspended" -> Presence(PresenceState.PAUSED, "Paused", if (task.resume == "ask") "Waiting for you to resume" else "Resumes when next in priority")
        else -> {
            val action = thread?.tools?.lastOrNull()
            val name = action?.name?.lowercase().orEmpty()
            when {
                action?.status == "failed" -> Presence(PresenceState.RECOVERING, "Recovering", "failed; choosing what to do next", action.name)
                action?.status == "running" && Regex("navigate|search|move_to|go_to|explore|walk|approach").containsMatchIn(name) -> Presence(PresenceState.NAVIGATING, "Navigating", "moving through the space", action.name)
                action?.status == "running" && Regex("look|observe|inspect|camera|view|perceive|scan").containsMatchIn(name) -> Presence(PresenceState.OBSERVING, "Observing", "looking with the head camera", action.name)
                action?.status == "running" && Regex("verify|finish|check").containsMatchIn(name) -> Presence(PresenceState.VERIFYING, "Verifying", "checking the result", action.name)
                action?.status == "running" -> Presence(PresenceState.INTERACTING, "Acting", "running a tool", action.name)
                else -> Presence(PresenceState.THINKING, "Thinking", "Planning the next action")
            }
        }
    }
}
