package dev.dream.companion.ui

import android.provider.Settings
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.dream.companion.data.Presence
import dev.dream.companion.data.PresenceState
import dev.dream.companion.data.Task
import kotlin.math.cos
import kotlin.math.sin

/** True when the system animator scale is 0 (Android's "remove animations") or the app toggle is on. */
@Composable
fun systemReducedMotion(): Boolean {
    val ctx = LocalContext.current
    return remember { Settings.Global.getFloat(ctx.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f }
}

fun tintFor(state: PresenceState): Color = when (state) {
    PresenceState.RECOVERING, PresenceState.FAILED -> D.amber
    PresenceState.SAFETY_STOPPED -> D.danger
    PresenceState.COMPLETED -> D.success
    PresenceState.OFFLINE -> D.text3
    PresenceState.LISTENING -> D.text
    else -> D.accent
}

/** "observe_scene · looking with the head camera · inspect cabinet": the tool as code, the rest in plain words. */
@Composable
fun PresenceLine(p: Presence, task: Task?, color: Color = D.text2, maxLines: Int = 2) {
    val active = task != null && p.state !in setOf(PresenceState.IDLE, PresenceState.OFFLINE, PresenceState.LISTENING)
    Text(
        buildAnnotatedString {
            p.tool?.let { withStyle(SpanStyle(fontFamily = GeistMono, color = D.text)) { append(it) }; append(" · ") }
            append(p.detail)
            if (active) append(" · ${task!!.text}")
        },
        style = T.small.copy(color = color), maxLines = maxLines,
    )
}

fun stateTone(state: String): Color = when (state) {
    "running" -> D.accent; "succeeded" -> D.success; "failed" -> D.danger; "suspended", "cancelled" -> D.amber; else -> D.text2
}
val STATE_LABEL = mapOf("queued" to "Queued", "running" to "Running", "suspended" to "Suspended", "succeeded" to "Runtime reports success", "failed" to "Failed", "cancelled" to "Cancelled")

/** Task state: a dot plus a written label, never colour alone. */
@Composable
fun StateChip(state: String, label: String? = null) {
    val tone = stateTone(state)
    Row(Modifier.clip(RoundedCornerShape(50)).background(tone.copy(alpha = 0.12f)).padding(horizontal = 8.dp, vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(6.dp).clip(CircleShape).background(tone))
        Text(label ?: STATE_LABEL[state] ?: state, style = T.label.copy(color = tone), modifier = Modifier.padding(start = 6.dp))
    }
}

@Composable
fun Pill(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, tone: Color = D.text, bg: Color = D.veilStrong, enabled: Boolean = true, leading: (@Composable () -> Unit)? = null) {
    Row(
        modifier.clip(RoundedCornerShape(10.dp)).background(if (enabled) bg else D.veil).clickable(enabled = enabled, onClick = onClick).height(36.dp).padding(horizontal = 14.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(7.dp),
    ) {
        leading?.invoke()
        Text(text, style = T.label.copy(color = if (enabled) tone else D.text3, fontSize = T.small.fontSize))
    }
}

@Composable
fun Section(label: String, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    androidx.compose.foundation.layout.Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(label, style = T.label)
        content()
    }
}

@Composable
fun Surface(modifier: Modifier = Modifier, padding: PaddingValues = PaddingValues(14.dp), content: @Composable () -> Unit) {
    Box(modifier.clip(RoundedCornerShape(16.dp)).background(D.card).border(1.dp, D.hairline, RoundedCornerShape(16.dp)).padding(padding)) { content() }
}

@Composable
fun Dot(color: Color, modifier: Modifier = Modifier) = Box(modifier.size(7.dp).clip(CircleShape).background(color))

@Composable
fun FullBleed(content: @Composable () -> Unit) = Box(Modifier.fillMaxSize()) { content() }

/** A round icon button whose glyph is a centred vector drawable. */
@Composable
fun IconCircle(icon: Int, description: String, onClick: () -> Unit, modifier: Modifier = Modifier, size: Dp = 40.dp, tint: Color = D.text, bg: Color = D.veilStrong, enabled: Boolean = true) {
    Box(
        modifier.size(size).clip(CircleShape).background(bg).clickable(enabled = enabled, onClick = onClick).semantics { contentDescription = description },
        contentAlignment = Alignment.Center,
    ) { androidx.compose.material3.Icon(androidx.compose.ui.res.painterResource(icon), contentDescription = null, tint = if (enabled) tint else D.text3, modifier = Modifier.size(size * 0.5f)) }
}

/** Full-width action button with an optional leading icon, content centred. */
@Composable
fun ActionButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, icon: Int? = null, tone: Color = D.text, bg: Color = D.veilStrong, enabled: Boolean = true) {
    Row(
        modifier.height(44.dp).clip(RoundedCornerShape(12.dp)).background(if (enabled) bg else D.veil)
            .clickable(enabled = enabled, role = androidx.compose.ui.semantics.Role.Button, onClick = onClick).semantics(mergeDescendants = true) { contentDescription = text }.padding(horizontal = 14.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.Center,
    ) {
        if (icon != null) androidx.compose.material3.Icon(androidx.compose.ui.res.painterResource(icon), contentDescription = null, tint = if (enabled) tone else D.text3, modifier = Modifier.size(16.dp))
        if (icon != null) androidx.compose.foundation.layout.Spacer(Modifier.size(8.dp))
        Text(text, style = T.label.copy(color = if (enabled) tone else D.text3, fontSize = T.small.fontSize), maxLines = 1)
    }
}

/** Two-option segmented control with equal halves. */
@Composable
fun Segmented(options: List<String>, selected: Int, onSelect: (Int) -> Unit, modifier: Modifier = Modifier) {
    Row(modifier.height(36.dp).clip(RoundedCornerShape(10.dp)).background(D.veilStrong).padding(3.dp)) {
        options.forEachIndexed { i, o ->
            Box(Modifier.fillMaxHeight().clip(RoundedCornerShape(8.dp)).background(if (i == selected) D.raised else Color.Transparent).clickable { onSelect(i) }.padding(horizontal = 12.dp), contentAlignment = Alignment.Center) {
                Text(o, style = T.label.copy(color = if (i == selected) D.text else D.text2))
            }
        }
    }
}

/** Section heading with an optional right-aligned note, both on one baseline. */
@Composable
fun Heading(title: String, note: String? = null) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(title, style = T.label.copy(color = D.text), modifier = Modifier.weight(1f))
        note?.let { Text(it, style = T.label.copy(color = D.text3)) }
    }
}

/** A labelled value tile; values are single-line so tiles in a row stay the same height. */
@Composable
fun Stat(label: String, value: String, modifier: Modifier = Modifier, color: Color = D.text) {
    androidx.compose.foundation.layout.Column(modifier.clip(RoundedCornerShape(12.dp)).background(D.veil).padding(horizontal = 12.dp, vertical = 10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text(label, style = T.label.copy(color = D.text3), maxLines = 1)
        Text(value, style = T.mono.copy(color = color), maxLines = 1, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis)
    }
}
