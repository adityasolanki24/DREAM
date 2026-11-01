package dev.dream.companion.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import dev.dream.companion.audio.MicState
import dev.dream.companion.data.SessionState
import dev.dream.companion.data.Task
import dev.dream.companion.data.Thread
import dev.dream.companion.data.argSummary
import dev.dream.companion.data.endTitle
import dev.dream.companion.data.presenceFor
import kotlinx.coroutines.launch

/** Everything the composer needs from the activity: draft, model sheet, voice and permission plumbing. */
class ComposerHooks(
    val draft: String, val setDraft: (String) -> Unit, val send: suspend (String) -> Boolean,
    val mic: MicState, val micPartial: String, val micError: String, val micLevel: Float,
    val startMic: () -> Unit, val stopMic: () -> Unit, val openModels: () -> Unit, val modelLabel: String,
)

@Composable
fun CompanionScreen(s: SessionState, hooks: ComposerHooks, reduced: Boolean, onAct: (String, String) -> Unit, onOpenMission: () -> Unit) {
    val shown = s.shown
    val thread = shown?.let { s.feed.threads[it.id] }
    val listening = hooks.mic == MicState.LISTENING || hooks.mic == MicState.STARTING
    val presence = presenceFor(shown, thread, s.snapshot?.stopReason, s.link, listening, shown?.let { s.endedAt[it.id] })
    val hasConversation = s.tasks.isNotEmpty()
    val list = rememberLazyListState()
    LaunchedEffect(s.tasks.size, thread?.tools?.size) { if (s.tasks.isNotEmpty()) list.animateScrollToItem(s.tasks.size) }

    Column(Modifier.fillMaxSize()) {
        // Presence: large and centred before any conversation, compact once there is one.
        Column(
            Modifier.fillMaxWidth().animateContentSize().padding(horizontal = 20.dp, vertical = if (hasConversation) 6.dp else 0.dp),
            horizontalAlignment = if (hasConversation) Alignment.Start else Alignment.CenterHorizontally,
        ) {
            if (!hasConversation) Spacer(Modifier.heightIn(min = 36.dp))
            if (hasConversation) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(56.dp), contentAlignment = Alignment.Center) { Orb(presence.state, 56.dp, reduced, hooks.micLevel) }
                    Column(Modifier.padding(start = 12.dp).weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(presence.title, style = T.title)
                        PresenceLine(presence, if (shown?.runnable == true) shown else null, maxLines = 2)
                    }
                }
            } else {
                Orb(presence.state, 168.dp, reduced, hooks.micLevel)
                Spacer(Modifier.width(1.dp).heightIn(min = 18.dp))
                Text(if (presence.state.name == "IDLE") "Ready when you are" else presence.title, style = T.display, textAlign = TextAlign.Center)
                Spacer(Modifier.heightIn(min = 6.dp))
                PresenceLine(presence, shown)
            }
            if (shown?.runnable == true) TaskControls(shown, s.stopping.contains(shown.id), s.link.name == "ONLINE", onAct, Modifier.padding(top = 10.dp))
        }

        if (hasConversation) {
            LazyColumn(
                Modifier.weight(1f).fillMaxWidth().padding(top = 8.dp)
                    .graphicsLayer { compositingStrategy = androidx.compose.ui.graphics.CompositingStrategy.Offscreen }
                    .drawWithContent {
                        drawContent()
                        drawRect(Brush.verticalGradient(0f to androidx.compose.ui.graphics.Color.Transparent, 28.dp.toPx() / size.height to androidx.compose.ui.graphics.Color.Black), blendMode = androidx.compose.ui.graphics.BlendMode.DstIn)
                    },
                state = list, contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 20.dp, bottom = 12.dp), verticalArrangement = Arrangement.spacedBy(22.dp),
            ) {
                items(s.tasks, key = { it.id }) { t -> Exchange(t, s.feed.threads[t.id], s.stopping.contains(t.id), null) }
                item { Spacer(Modifier.heightIn(min = 4.dp)) }
            }
        } else {
            Spacer(Modifier.weight(1f))
            Text(
                if (s.link.name == "ONLINE") "Hold the microphone to speak a goal, or type one. Nothing is sent until you press send."
                else "Connect in Settings to send goals to DREAM.",
                style = T.small, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth().padding(horizontal = 36.dp, vertical = 10.dp),
            )
        }
        Composer(s, hooks)
    }
}

fun modelLabel(s: SessionState, t: Task): String? = when {
    t.provider == null -> null
    t.provider == "structured" -> "Structured parser"
    t.model != null -> t.model
    else -> s.choices.find { it.id == t.provider }?.label ?: t.provider
}

@Composable
fun TaskControls(task: Task, stopping: Boolean, online: Boolean, onAct: (String, String) -> Unit, modifier: Modifier = Modifier) {
    Row(modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        if (task.state == "suspended") ActionButton("Resume", { onAct(task.id, "resume") }, Modifier.weight(1f), dev.dream.companion.R.drawable.ic_play, enabled = online)
        else ActionButton(if (stopping) "Pausing…" else "Pause", { onAct(task.id, "pause") }, Modifier.weight(1f), dev.dream.companion.R.drawable.ic_pause, enabled = online && task.state == "running" && !stopping)
        // Cancel stays enabled whatever the animations or voice are doing; it is the operator's interruption.
        ActionButton(if (stopping) "Stopping at safe point…" else "Cancel", { onAct(task.id, "cancel") }, Modifier.weight(1f), dev.dream.companion.R.drawable.ic_stop, tone = D.danger, bg = D.danger.copy(alpha = 0.14f), enabled = online && !stopping)
    }
}

@Composable
fun Exchange(t: Task, thread: Thread?, stopping: Boolean, model: String?, compact: Boolean = false) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.CenterEnd) {
            Text(t.text, style = T.body, modifier = Modifier.clip(RoundedCornerShape(18.dp, 18.dp, 6.dp, 18.dp)).background(D.raised).padding(horizontal = 14.dp, vertical = 9.dp))
        }
        val state = if (t.runnable) t.state else thread?.endState ?: t.state
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(14.dp).clip(CircleShape).background(Brush.radialGradient(listOf(D.text, D.accent, androidx.compose.ui.graphics.Color(0xFF2B3A4C)))))
            Text("DREAM", style = T.label.copy(color = D.text), modifier = Modifier.padding(start = 8.dp))
            model?.let { Text(it, style = T.monoSmall.copy(color = D.text3), maxLines = 1, modifier = Modifier.padding(start = 8.dp).weight(1f, false)) }
            Spacer(Modifier.weight(1f))
            StateChip(state)
        }
        if (stopping && t.runnable) Text("Waiting for next safe point", style = T.small.copy(color = D.accent), modifier = Modifier.padding(start = 22.dp))
        val tools = thread?.tools.orEmpty().let { if (compact) it.takeLast(3) else it }
        Column(Modifier.padding(start = 4.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) { tools.forEach { ToolRow(it) } }
        if (t.state == "queued") Text("Waiting for the robot to be free.", style = T.small, modifier = Modifier.padding(start = 22.dp))
        if (state == "running" && !stopping && tools.isEmpty()) Text("Planning the first action…", style = T.small, modifier = Modifier.padding(start = 22.dp))
        if (state in setOf("succeeded", "failed", "cancelled")) {
            val tone = when (state) { "succeeded" -> D.success; "cancelled" -> D.amber; else -> D.danger }
            Column(Modifier.padding(start = 22.dp).fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(D.veil).padding(12.dp)) {
                Text("${endTitle(state)}. ${thread?.endReason ?: t.result.orEmpty()}", style = T.small.copy(color = D.text))
                Text(when (state) { "succeeded" -> "The runtime's own report; no independent check is shown here."; "cancelled" -> "The task was stopped and was not completed."; else -> "" },
                    style = T.monoSmall.copy(fontFamily = Geist, color = tone))
            }
        }
        if (state == "suspended") Text(if (t.resume == "ask") "Paused by you · waiting for Resume" else "Paused for interrupt · resumes automatically when next in priority", style = T.small.copy(color = D.amber), modifier = Modifier.padding(start = 22.dp))
    }
}

@Composable
fun ToolRow(tool: dev.dream.companion.data.Tool) {
    val tone = when (tool.status) { "succeeded" -> D.success; "running" -> D.accent; "interrupted" -> D.amber; else -> D.danger }
    Row(verticalAlignment = Alignment.Top) {
        Box(Modifier.padding(top = 5.dp).size(8.dp).clip(CircleShape).background(if (tool.status == "running") tone.copy(alpha = 0.5f) else tone))
        Column(Modifier.padding(start = 12.dp)) {
            Row { Text(tool.name, style = T.mono); val a = argSummary(tool.args); if (a.isNotEmpty()) Text("  $a", style = T.mono.copy(color = D.text2), maxLines = 2) }
            val note = if (tool.status != "succeeded" && tool.status != "running") "${tool.status}${if (tool.message.isNotEmpty()) ": ${tool.message}" else ""}" else tool.message
            if (note.isNotEmpty()) Text(note, style = T.small.copy(color = if (tool.status == "succeeded" || tool.status == "running") D.text2 else tone))
        }
    }
}

@Composable
fun Composer(s: SessionState, h: ComposerHooks) {
    val scope = rememberCoroutineScope()
    var busy by remember { mutableStateOf(false) }
    val online = s.link.name == "ONLINE"
    val examples = s.choices.find { it.id == (s.choice?.provider ?: s.snapshot?.provider) }?.supports.orEmpty()
    fun send() { val text = h.draft.trim(); if (text.isEmpty() || busy || !online) return; busy = true; scope.launch { if (h.send(text)) h.setDraft(""); busy = false } }

    Column(Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, bottom = 10.dp, top = 4.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (s.notice.isNotEmpty()) Text(s.notice, style = T.small.copy(color = D.amber), modifier = Modifier.padding(horizontal = 6.dp))
        AnimatedVisibility(online && h.draft.isEmpty() && examples.isNotEmpty() && h.mic == MicState.IDLE, enter = fadeIn(), exit = fadeOut()) {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.horizontalScroll(androidx.compose.foundation.rememberScrollState()).padding(horizontal = 4.dp)) {
                examples.forEach { x -> Text(x, style = T.small, modifier = Modifier.clip(RoundedCornerShape(50)).border(1.dp, D.hairline, RoundedCornerShape(50)).clickable { h.setDraft(if ('<' in x) x.substringBefore('<') else x) }.padding(horizontal = 12.dp, vertical = 6.dp)) }
            }
        }
        val micStatus = when (h.mic) {
            MicState.STARTING -> "Starting microphone…"
            MicState.LISTENING -> if (h.micPartial.isNotEmpty()) h.micPartial else "Listening — release, or tap the mic, to finish"
            MicState.PROCESSING -> "Transcribing…"
            MicState.UNAVAILABLE -> "Speech recognition is not available on this phone"
            MicState.ERROR -> h.micError
            MicState.IDLE -> h.micError.takeIf { it.isNotEmpty() }
        }
        micStatus?.let { Text(it, style = T.small.copy(color = if (h.mic == MicState.LISTENING) D.text else D.text2), modifier = Modifier.padding(horizontal = 6.dp), maxLines = 2) }
        Column {
            Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(if (online) D.card else D.veil).border(1.dp, if (h.mic == MicState.LISTENING) D.accent.copy(alpha = 0.6f) else D.hairline, RoundedCornerShape(22.dp)).padding(start = 16.dp, end = 6.dp, top = 12.dp, bottom = 6.dp)) {
                BasicTextField(
                    value = h.draft, onValueChange = { h.setDraft(it.take(500)) }, enabled = online, textStyle = T.body,
                    cursorBrush = SolidColor(D.accent), keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send), keyboardActions = KeyboardActions(onSend = { send() }),
                    modifier = Modifier.fillMaxWidth().heightIn(min = 24.dp, max = 120.dp).semantics { contentDescription = "Task" },
                    decorationBox = { inner -> Box { if (h.draft.isEmpty()) Text(if (online) "Ask DREAM to ${examples.firstOrNull()?.substringBefore('<')?.trim() ?: "do something"}…" else "Connect to send a task", style = T.body.copy(color = D.text3)); inner() } },
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("${h.modelLabel}  ▾", style = T.label, modifier = Modifier.clip(RoundedCornerShape(8.dp)).clickable(enabled = online) { h.openModels() }.padding(horizontal = 4.dp, vertical = 8.dp))
                    Spacer(Modifier.weight(1f))
                    MicButton(h, online)
                    Spacer(Modifier.width(8.dp))
                    val ready = online && h.draft.isNotBlank() && !busy
                    Box(Modifier.size(36.dp).clip(CircleShape).background(if (ready) D.text else D.veilStrong).clickable(enabled = ready) { send() }.semantics { contentDescription = "Send task" }, contentAlignment = Alignment.Center) {
                        if (busy) CircularProgressIndicator(Modifier.size(16.dp), color = D.page, strokeWidth = 2.dp)
                        else androidx.compose.material3.Icon(androidx.compose.ui.res.painterResource(dev.dream.companion.R.drawable.ic_arrow_up), null, tint = if (ready) D.page else D.text3, modifier = Modifier.size(18.dp))
                    }
                }
            }
        }
    }
}

/** Push-to-talk: hold to speak, release to finish. A quick tap toggles listening on, a second tap stops (accessibility). */
@Composable
fun MicButton(h: ComposerHooks, online: Boolean) {
    var toggled by remember { mutableStateOf(false) }
    var pressedAt by remember { mutableLongStateOf(0L) }
    val active = h.mic == MicState.LISTENING || h.mic == MicState.STARTING
    val usable = online && h.mic != MicState.UNAVAILABLE
    val scale by animateFloatAsState(if (active) 1.08f + h.micLevel * 0.12f else 1f, label = "mic")
    val label = when { !usable -> "Voice input unavailable"; active -> "Stop listening"; else -> "Hold to talk, or tap to start listening" }
    Box(
        Modifier.size(36.dp).scale(scale).clip(CircleShape)
            .background(if (active) Brush.linearGradient(listOf(D.cyan, D.blue, D.violet)) else Brush.linearGradient(listOf(D.veilStrong, D.veilStrong)))
            .semantics { contentDescription = label }
            .pointerInput(usable, toggled) {
                if (!usable) return@pointerInput
                detectTapGestures(onPress = {
                    if (toggled) { toggled = false; h.stopMic(); return@detectTapGestures }
                    pressedAt = System.currentTimeMillis()
                    h.startMic()
                    tryAwaitRelease()
                    if (System.currentTimeMillis() - pressedAt < 350) toggled = true else h.stopMic()
                })
            },
        contentAlignment = Alignment.Center,
    ) {
        if (h.mic == MicState.PROCESSING) CircularProgressIndicator(Modifier.size(16.dp), color = D.text, strokeWidth = 2.dp)
        else androidx.compose.material3.Icon(androidx.compose.ui.res.painterResource(dev.dream.companion.R.drawable.ic_mic), null, tint = if (active) D.page else if (usable) D.text else D.text3, modifier = Modifier.size(18.dp))
    }
    LaunchedEffect(h.mic) { if (h.mic != MicState.LISTENING && h.mic != MicState.STARTING) toggled = false }
}

@Composable
fun MicGlyph(color: androidx.compose.ui.graphics.Color) {
    androidx.compose.foundation.Canvas(Modifier.size(24.dp)) {
        val w = size.width; val s = androidx.compose.ui.graphics.drawscope.Stroke(2.dp.toPx(), cap = androidx.compose.ui.graphics.StrokeCap.Round)
        drawRoundRect(color, topLeft = androidx.compose.ui.geometry.Offset(w * 0.35f, w * 0.08f), size = androidx.compose.ui.geometry.Size(w * 0.3f, w * 0.52f), cornerRadius = androidx.compose.ui.geometry.CornerRadius(w * 0.15f))
        drawArc(color, 0f, 180f, false, topLeft = androidx.compose.ui.geometry.Offset(w * 0.22f, w * 0.28f), size = androidx.compose.ui.geometry.Size(w * 0.56f, w * 0.48f), style = s)
        drawLine(color, androidx.compose.ui.geometry.Offset(w / 2, w * 0.76f), androidx.compose.ui.geometry.Offset(w / 2, w * 0.92f), 2.dp.toPx(), androidx.compose.ui.graphics.StrokeCap.Round)
    }
}
