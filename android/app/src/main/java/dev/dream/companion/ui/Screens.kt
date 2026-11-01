package dev.dream.companion.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import dev.dream.companion.camera.CameraState
import dev.dream.companion.camera.CameraStatus
import dev.dream.companion.data.Link
import dev.dream.companion.data.ModelChoice
import dev.dream.companion.data.SessionState
import dev.dream.companion.data.Snapshot
import dev.dream.companion.data.argSummary
import dev.dream.companion.data.presenceFor
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/* ---------------------------------------------------------------- Live */

fun cameraLabel(c: CameraState, online: Boolean): String = when {
    !online -> "Disconnected · last frame"
    c.status == CameraStatus.LIVE -> "${c.transport} · ${"%.0f".format(c.fps)} fps"
    c.status == CameraStatus.PAUSED -> "Simulation paused or not sensing"
    c.status == CameraStatus.STALE -> "Stale frame"
    c.status == CameraStatus.NO_FRAMES -> "No camera frames yet"
    c.status == CameraStatus.UNAVAILABLE -> "Camera unavailable"
    c.status == CameraStatus.ERROR -> c.message.ifEmpty { "Camera error" }
    else -> "Connecting to camera…"
}

fun cameraTone(c: CameraState, online: Boolean): Color = when {
    !online -> D.amber
    c.status == CameraStatus.LIVE -> D.success
    c.status == CameraStatus.PAUSED || c.status == CameraStatus.STALE -> D.amber
    else -> D.text3
}

/** The video itself, with nothing drawn over it except a centred note when there is no frame. */
@Composable
fun CameraView(c: CameraState, online: Boolean, modifier: Modifier = Modifier, fit: Boolean = false) {
    Box(modifier.background(Color(0xFF070809)), contentAlignment = Alignment.Center) {
        val frame = c.frame
        if (frame != null) {
            Image(frame.asImageBitmap(), contentDescription = "Robot head camera frame", contentScale = if (fit) ContentScale.Fit else ContentScale.Crop,
                modifier = Modifier.fillMaxSize(), alpha = if (!online) 0.45f else if (c.status == CameraStatus.STALE) 0.75f else 1f)
        } else {
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(28.dp)) {
                Text(if (online) "Waiting for the first frame" else "Not connected", style = T.title, textAlign = TextAlign.Center)
                Text(if (online) "Frames come from the robot's head-camera stream, or with each observation a task makes." else "Connect to see what the robot sees.",
                    style = T.small, textAlign = TextAlign.Center)
            }
        }
    }
}

@Composable
fun CameraStatusRow(c: CameraState, online: Boolean, modifier: Modifier = Modifier) {
    Row(modifier, verticalAlignment = Alignment.CenterVertically) {
        Dot(cameraTone(c, online))
        Text(cameraLabel(c, online), style = T.label.copy(color = D.text), maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(start = 8.dp).weight(1f, fill = false))
        if (c.frame != null) c.ageS?.let { Text("  ${if (it < 10) "%.1f".format(it) else "%.0f".format(it)} s old", style = T.label.copy(color = D.text3), maxLines = 1) }
    }
}

@Composable
fun LiveScreen(s: SessionState, cam: CameraState, reduced: Boolean, onAct: (String, String) -> Unit, onFullscreen: () -> Unit, listening: Boolean) {
    val online = s.link == Link.ONLINE
    val shown = s.shown
    val presence = presenceFor(shown, shown?.let { s.feed.threads[it.id] }, s.snapshot?.stopReason, s.link, listening, shown?.let { s.endedAt[it.id] })
    var view by remember { mutableStateOf(0) }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 4.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Box(Modifier.fillMaxWidth().aspectRatio(4f / 3f).clip(RoundedCornerShape(18.dp)).border(1.dp, if (s.snapshot?.stopReason != null) D.danger else D.hairline, RoundedCornerShape(18.dp))) {
            if (view == 1) MapView(s.snapshot, Modifier.fillMaxSize(), showLegend = false) else CameraView(cam, online, Modifier.fillMaxSize().clickable(enabled = cam.frame != null) { onFullscreen() })
        }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            if (view == 0) CameraStatusRow(cam, online, Modifier.weight(1f)) else Text("Spatial map · this session", style = T.label.copy(color = D.text), modifier = Modifier.weight(1f))
            Spacer(Modifier.width(8.dp))
            Segmented(listOf("Camera", "Map"), view, { view = it })
            Spacer(Modifier.width(8.dp))
            IconCircle(dev.dream.companion.R.drawable.ic_fullscreen, "Full screen camera", onFullscreen, size = 36.dp, enabled = cam.frame != null)
        }
        ActivityCard(presence, shown, s, online, reduced, onAct)
        s.snapshot?.let { snap ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Stat("Robot clock", "%.1f s".format(snap.robotTime), Modifier.weight(1f))
                Stat("x, y (m)", if (snap.odometry.size >= 2) "%.1f, %.1f".format(snap.odometry[0], snap.odometry[1]) else "—", Modifier.weight(1f))
                Stat("Safety stop", snap.stopReason ?: "none", Modifier.weight(1f), if (snap.stopReason != null) D.danger else D.text)
            }
            Text("${snap.environment ?: "Environment not labelled"} · dropped frames ${cam.dropped}", style = T.monoSmall, modifier = Modifier.padding(horizontal = 4.dp))
        }
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
fun ActivityCard(presence: dev.dream.companion.data.Presence, shown: dev.dream.companion.data.Task?, s: SessionState, online: Boolean, reduced: Boolean, onAct: (String, String) -> Unit) {
    Surface(Modifier.fillMaxWidth(), padding = PaddingValues(16.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(48.dp), contentAlignment = Alignment.Center) { Orb(presence.state, 48.dp, reduced) }
                Column(Modifier.padding(start = 12.dp).weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(presence.title, style = T.title)
                    PresenceLine(presence, if (shown?.runnable == true) shown else null)
                }
            }
            if (shown?.runnable == true) TaskControls(shown, s.stopping.contains(shown.id), online, onAct)
        }
    }
}

@Composable
fun FullscreenCamera(cam: CameraState, online: Boolean, onClose: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Color.Black)) {
        CameraView(cam, online, Modifier.fillMaxSize(), fit = true)
        Row(Modifier.align(Alignment.TopStart).fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Row(Modifier.weight(1f).clip(RoundedCornerShape(50)).background(D.glass).padding(horizontal = 12.dp, vertical = 8.dp)) { CameraStatusRow(cam, online) }
            Spacer(Modifier.width(12.dp))
            IconCircle(dev.dream.companion.R.drawable.ic_close, "Close full screen", onClose, size = 40.dp, bg = D.glass)
        }
    }
}

/* ---------------------------------------------------------------- Map and world */

/** Named candidate colours from the perception pipeline, drawn as they are reported. */
fun candidateColor(name: String?): Color = when (name?.lowercase()) {
    "red" -> Color(0xFFE5675A); "green" -> Color(0xFF5DBB7A); "blue" -> Color(0xFF5C8DE8); "yellow" -> Color(0xFFE6C24F)
    "orange" -> Color(0xFFE8964A); "purple" -> Color(0xFF9B74E0); "white" -> Color(0xFFE8E6E0); "black" -> Color(0xFF3A3D42)
    else -> D.amber
}

/** Distance and plain-language direction of a point relative to the robot's pose (x, y, yaw); null without a pose. */
fun relativeTo(odo: List<Double>, x: Double, y: Double): Pair<Double, String>? {
    if (odo.size < 3) return null
    val dx = x - odo[0]; val dy = y - odo[1]
    var rel = Math.toDegrees(kotlin.math.atan2(dy, dx) - odo[2])
    while (rel > 180) rel -= 360
    while (rel < -180) rel += 360
    val dir = when {
        kotlin.math.abs(rel) < 22.5 -> "ahead"
        rel in 22.5..67.5 -> "ahead-left"
        rel in -67.5..-22.5 -> "ahead-right"
        rel in 67.5..112.5 -> "left"
        rel in -112.5..-67.5 -> "right"
        rel > 112.5 && rel < 157.5 -> "behind-left"
        rel < -112.5 && rel > -157.5 -> "behind-right"
        else -> "behind"
    }
    return kotlin.math.hypot(dx, dy) to dir
}

@Composable
fun MapView(snap: Snapshot?, modifier: Modifier = Modifier, showLegend: Boolean = true) {
    val map = snap?.map
    val measurer = rememberTextMeasurer()
    Column(modifier.background(Color(0xFF111316))) {
        Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            if (map == null || map.cells.isEmpty()) { Text("No map yet · built from what the robot observes", style = T.small, textAlign = TextAlign.Center, modifier = Modifier.padding(24.dp)); return@Box }
            Canvas(Modifier.fillMaxSize().padding(12.dp)) {
                val cols = map.cells.size; val rows = map.cells[0].size
                val cell = minOf(size.width / cols, size.height / rows)
                val ox = (size.width - cell * cols) / 2; val oy = (size.height - cell * rows) / 2
                for (x in 0 until cols) for (y in 0 until rows) {
                    val v = map.cells[x][y]; if (v == 0) continue
                    drawRect(if (v == 2) Color(0xFFD9D5CC) else Color(0xFF2B2F35), Offset(ox + x * cell, oy + (rows - 1 - y) * cell), androidx.compose.ui.geometry.Size(cell + 0.6f, cell + 0.6f))
                }
                fun px(wx: Double, wy: Double) = Offset(ox + ((wx - map.origin[0]) / map.resolution).toFloat() * cell, oy + (rows - ((wy - map.origin[1]) / map.resolution).toFloat()) * cell)
                val badge = 11.dp.toPx()
                snap.objects.forEach { o ->
                    val p = px(o.x, o.y)
                    drawCircle(Color(0xFF111316), badge + 2.dp.toPx(), p)
                    drawCircle(candidateColor(o.color), badge, p)
                    val t = measurer.measure("${o.id}", T.label.copy(color = Color(0xFF111316)))
                    drawText(t, topLeft = Offset(p.x - t.size.width / 2f, p.y - t.size.height / 2f))
                }
                if (snap.odometry.size >= 3) {
                    val p = px(snap.odometry[0], snap.odometry[1]); val yaw = snap.odometry[2]
                    drawCircle(D.accent.copy(alpha = 0.22f), 16.dp.toPx(), p)
                    rotate(-Math.toDegrees(yaw).toFloat(), p) {
                        val path = androidx.compose.ui.graphics.Path().apply {
                            moveTo(p.x + 10.dp.toPx(), p.y); lineTo(p.x - 7.dp.toPx(), p.y + 7.dp.toPx()); lineTo(p.x - 3.dp.toPx(), p.y); lineTo(p.x - 7.dp.toPx(), p.y - 7.dp.toPx()); close()
                        }
                        drawPath(path, D.accent)
                    }
                }
            }
        }
        if (showLegend && map != null) Row(Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            LegendItem(Color(0xFF2B2F35), "free"); LegendItem(Color(0xFFD9D5CC), "occupied"); LegendItem(D.accent, "robot")
            Spacer(Modifier.weight(1f))
            Text("${map.resolution} m/cell", style = T.monoSmall)
        }
    }
}

@Composable
fun LegendItem(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically) { Box(Modifier.size(9.dp).clip(RoundedCornerShape(2.dp)).background(color)); Text(label, style = T.label, modifier = Modifier.padding(start = 6.dp)) }
}

/**
 * Robot-centred view of what has been seen: the robot at the centre facing up, each candidate placed by its measured
 * distance and bearing, with the edge labelled. Derived only from reported positions and the (oracle-assisted) pose;
 * the backend publishes no semantic relations, so none are invented.
 */
@Composable
fun RelationsView(snap: Snapshot, modifier: Modifier = Modifier) {
    val measurer = rememberTextMeasurer()
    val rel = snap.objects.mapNotNull { o -> relativeTo(snap.odometry, o.x, o.y)?.let { Triple(o, it.first, kotlin.math.atan2(o.y - snap.odometry[1], o.x - snap.odometry[0]) - snap.odometry[2]) } }
    val maxD = (rel.maxOfOrNull { it.second } ?: 1.0).coerceAtLeast(1.0)
    val ringStep = if (maxD > 4) 2.0 else 1.0
    Canvas(modifier) {
        val c = center; val radius = size.minDimension / 2 - 22.dp.toPx()
        val scale = radius / (kotlin.math.ceil(maxD / ringStep) * ringStep).toFloat()
        var r = ringStep
        while (r * scale <= radius + 1) {
            drawCircle(D.hairline, (r * scale).toFloat(), c, style = Stroke(1.dp.toPx()))
            val t = measurer.measure("${r.toInt()} m", T.monoSmall.copy(color = D.text3))
            drawText(t, topLeft = Offset(c.x + 4.dp.toPx(), c.y - (r * scale).toFloat() - t.size.height))
            r += ringStep
        }
        rel.forEach { (o, d, bearing) ->
            // screen: up is the robot's heading, left is +bearing
            val p = Offset(c.x - (kotlin.math.sin(bearing) * d * scale).toFloat(), c.y - (kotlin.math.cos(bearing) * d * scale).toFloat())
            drawLine(D.text3, c, p, 1.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(6f, 6f)))
            val mid = Offset((c.x + p.x) / 2, (c.y + p.y) / 2)
            val lt = measurer.measure("%.1f m".format(d), T.monoSmall.copy(color = D.text2))
            drawRoundRect(Color(0xFF111316), Offset(mid.x - lt.size.width / 2f - 4f, mid.y - lt.size.height / 2f), androidx.compose.ui.geometry.Size(lt.size.width + 8f, lt.size.height.toFloat()), androidx.compose.ui.geometry.CornerRadius(6f))
            drawText(lt, topLeft = Offset(mid.x - lt.size.width / 2f, mid.y - lt.size.height / 2f))
            drawCircle(candidateColor(o.color), 12.dp.toPx(), p)
            val t = measurer.measure("${o.id}", T.label.copy(color = Color(0xFF111316)))
            drawText(t, topLeft = Offset(p.x - t.size.width / 2f, p.y - t.size.height / 2f))
        }
        val path = androidx.compose.ui.graphics.Path().apply { moveTo(c.x, c.y - 11.dp.toPx()); lineTo(c.x + 8.dp.toPx(), c.y + 8.dp.toPx()); lineTo(c.x, c.y + 4.dp.toPx()); lineTo(c.x - 8.dp.toPx(), c.y + 8.dp.toPx()); close() }
        drawCircle(D.accent.copy(alpha = 0.2f), 18.dp.toPx(), c)
        drawPath(path, D.accent)
    }
}

@Composable
fun WorldScreen(s: SessionState) {
    val snap = s.snapshot
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 4.dp), verticalArrangement = Arrangement.spacedBy(20.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Heading("Spatial map", snap?.map?.let { "odometry ${it.odometry}" })
            MapView(snap, Modifier.fillMaxWidth().aspectRatio(0.95f).clip(RoundedCornerShape(18.dp)).border(1.dp, D.hairline, RoundedCornerShape(18.dp)))
        }
        val objects = snap?.objects.orEmpty()
        if (snap != null && objects.isNotEmpty() && snap.odometry.size >= 3) Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Heading("Around the robot", "from measured positions")
            Box(Modifier.fillMaxWidth().aspectRatio(1.1f).clip(RoundedCornerShape(18.dp)).background(Color(0xFF111316)).border(1.dp, D.hairline, RoundedCornerShape(18.dp))) {
                RelationsView(snap, Modifier.fillMaxSize().padding(8.dp))
                Text("Robot faces up", style = T.label.copy(color = D.text3), modifier = Modifier.align(Alignment.BottomCenter).padding(10.dp))
            }
        }
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Heading("Seen this session", "${objects.size} candidates · not persistent")
            if (objects.isEmpty()) Text("Objects appear after the robot looks around.", style = T.small, modifier = Modifier.padding(horizontal = 4.dp))
            objects.forEach { o ->
                val r = snap?.let { relativeTo(it.odometry, o.x, o.y) }
                Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(D.card).padding(horizontal = 14.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(30.dp).clip(CircleShape).background(candidateColor(o.color)), contentAlignment = Alignment.Center) { Text("${o.id}", style = T.label.copy(color = Color(0xFF111316))) }
                    Column(Modifier.padding(start = 12.dp).weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text("Candidate ${o.id}", style = T.body)
                        Text("${o.color ?: "colour unknown"} · seen ${o.views}×", style = T.small)
                    }
                    Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(r?.let { "%.1f m".format(it.first) } ?: "(%.1f, %.1f)".format(o.x, o.y), style = T.mono)
                        Text(r?.second ?: "map position", style = T.small)
                    }
                }
            }
        }
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Heading("Home memory", "read-only")
            when {
                snap == null -> Text("Not connected.", style = T.small, modifier = Modifier.padding(horizontal = 4.dp))
                !snap.memoryAvailable -> Text(snap.memoryError ?: "Start the backend with --memory <home-memory.sqlite> to show stored sightings.", style = T.small, modifier = Modifier.padding(horizontal = 4.dp))
                snap.memory.isEmpty() -> Text("Home memory is empty. ${snap.memorySource.orEmpty()}", style = T.small, modifier = Modifier.padding(horizontal = 4.dp))
                else -> snap.memory.forEach { m ->
                    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(D.card).padding(horizontal = 14.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) { Text(m.label, style = T.body); Text("${m.mapId} · ${m.source}", style = T.small) }
                        Text("(%.1f, %.1f)".format(m.x, m.y), style = T.mono)
                    }
                }
            }
        }
        Spacer(Modifier.height(8.dp))
    }
}

/* ---------------------------------------------------------------- Mission */

@Composable
fun MissionScreen(s: SessionState, onAct: (String, String) -> Unit) {
    var selected by remember { mutableStateOf<String?>(null) }
    val task = s.tasks.find { it.id == selected } ?: s.shown
    val thread = task?.let { s.feed.threads[it.id] }
    val online = s.link == Link.ONLINE
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 4.dp), verticalArrangement = Arrangement.spacedBy(20.dp)) {
        if (task == null) {
            Surface(Modifier.fillMaxWidth(), padding = PaddingValues(20.dp)) { Text("No tasks in this session yet. Send a goal from Companion.", style = T.small, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth()) }
            return@Column
        }
        Surface(Modifier.fillMaxWidth(), padding = PaddingValues(16.dp)) {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(if (task.runnable) "Current task" else "Task", style = T.label, modifier = Modifier.weight(1f))
                    StateChip(if (task.runnable) task.state else thread?.endState ?: task.state)
                }
                Text(task.text, style = T.title)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Stat("Steps", "${task.steps}", Modifier.weight(1f)); Stat("Priority", "${task.priority}", Modifier.weight(1f)); Stat("Pauses", "${task.suspensions}", Modifier.weight(1f))
                }
                modelLabel(s, task)?.let { Text("Parser · $it", style = T.small) }
                if (task.runnable) {
                    TaskControls(task, s.stopping.contains(task.id), online, onAct)
                    Text("Pause and cancel take effect at the robot's next safe boundary; the phone never commands motion directly.", style = T.small.copy(color = D.text3))
                }
            }
        }
        if (thread?.endState != null || thread?.claimSuccess != null) Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Heading("Verification")
            thread.endState?.let { Verify("Runtime outcome", it, thread.endReason, if (it == "succeeded") D.success else if (it == "cancelled") D.amber else D.danger) }
            thread.claimSuccess?.let { Verify("Agent finish claim · self-reported", if (it) "success" else "not achieved", thread.claimReason, if (it) D.success else D.danger) }
        }
        Column {
            Heading("Tool timeline", thread?.tools?.takeIf { it.isNotEmpty() }?.let { "${it.size} calls · observed, not a plan" })
            Spacer(Modifier.height(10.dp))
            if (thread?.tools.isNullOrEmpty()) Text("Tool calls appear here as the robot executes them.", style = T.small, modifier = Modifier.padding(horizontal = 4.dp))
            thread?.tools?.forEachIndexed { i, tool -> TimelineRow(i, tool, last = i == thread.tools.lastIndex) }
        }
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Heading("History", "${s.tasks.size}")
            s.tasks.reversed().forEach { t ->
                Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(if (t.id == task.id) D.accent.copy(alpha = 0.12f) else D.card).clickable { selected = t.id }.padding(horizontal = 14.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Dot(stateTone(t.state))
                    Column(Modifier.padding(start = 12.dp).weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(t.text, style = T.body, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text("${STATE_LABEL[t.state] ?: t.state} · ${t.steps} steps", style = T.small)
                    }
                    Text(SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date((t.updated * 1000).toLong())), style = T.mono.copy(color = D.text2))
                }
            }
        }
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
fun Verify(k: String, v: String, reason: String?, tone: Color) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(D.card).padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.width(3.dp).height(36.dp).clip(RoundedCornerShape(2.dp)).background(tone))
        Column(Modifier.padding(start = 12.dp)) { Text(k, style = T.label); Text("$v${reason?.let { " · $it" } ?: ""}", style = T.body.copy(color = D.text)) }
    }
}

/** One step of the observed tool sequence, drawn as a vertical stepper (status dot, rail, content). */
@Composable
fun TimelineRow(i: Int, tool: dev.dream.companion.data.Tool, last: Boolean) {
    var open by remember { mutableStateOf(false) }
    val tone = when (tool.status) { "succeeded" -> D.success; "running" -> D.accent; "interrupted" -> D.amber; else -> D.danger }
    Row(Modifier.fillMaxWidth().height(IntrinsicSize.Min)) {
        Column(Modifier.width(24.dp).fillMaxHeight(), horizontalAlignment = Alignment.CenterHorizontally) {
            Box(Modifier.padding(top = 18.dp).size(10.dp).clip(CircleShape).background(tone))
            if (!last) Box(Modifier.padding(top = 4.dp).width(1.dp).weight(1f).background(D.hairline))
        }
        Column(Modifier.weight(1f).padding(start = 8.dp, bottom = 8.dp).clip(RoundedCornerShape(14.dp)).background(D.card).clickable { open = !open }.padding(horizontal = 14.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("%02d".format(i + 1), style = T.monoSmall.copy(color = D.text3))
                Text(tool.name, style = T.mono, modifier = Modifier.padding(start = 10.dp).weight(1f), maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(tool.status, style = T.label.copy(color = tone))
            }
            val a = argSummary(tool.args)
            if (a.isNotEmpty()) Text(a, style = T.monoSmall)
            if (tool.message.isNotEmpty()) Text(tool.message, style = T.small)
            if (open) Text(tool.args.toString(2), style = T.monoSmall.copy(color = D.text), modifier = Modifier.padding(top = 4.dp).fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(D.page).padding(10.dp))
        }
    }
}

/* ---------------------------------------------------------------- Models */

@Composable
fun ModelSheet(s: SessionState, current: ModelChoice?, onPick: (ModelChoice) -> Unit, onClose: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Color(0x99000000)).clickable { onClose() }, contentAlignment = Alignment.BottomCenter) {
        Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(topStart = 22.dp, topEnd = 22.dp)).background(D.raised).clickable(enabled = false) {}.padding(20.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Model for new tasks", style = T.title)
            Text("Only providers this backend reports as available can be chosen.", style = T.small)
            s.choices.forEach { p ->
                Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(D.card).padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(p.label, style = T.body)
                    if (p.supports.isNotEmpty()) Text("Accepts: ${p.supports.joinToString(" · ")}", style = T.small)
                    if (!p.available) Text("Unavailable · ${p.reason ?: "not configured on this backend"}", style = T.small.copy(color = D.amber))
                    else {
                        val options = if (p.models.isEmpty()) listOf<String?>(null) else p.models
                        options.forEach { m ->
                            val choice = ModelChoice(p.id, m)
                            val sel = current == choice
                            Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(if (sel) D.accent.copy(alpha = 0.16f) else D.veil).clickable { onPick(choice); onClose() }.padding(10.dp)) {
                                Text(m ?: "Use ${p.label}", style = if (m != null) T.mono else T.small.copy(color = D.text), modifier = Modifier.weight(1f)); if (sel) Text("✓", style = T.body)
                            }
                        }
                    }
                }
            }
            if (s.choices.isEmpty()) Text("Connect to load providers.", style = T.small)
        }
    }
}

/* ---------------------------------------------------------------- Settings / connect */

@Composable
fun Field(label: String, value: String, onChange: (String) -> Unit, secret: Boolean = false) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(label, style = T.label)
        BasicTextField(value, onChange, singleLine = true, textStyle = T.mono.copy(fontSize = T.body.fontSize), cursorBrush = SolidColor(D.accent),
            visualTransformation = if (secret) PasswordVisualTransformation() else androidx.compose.ui.text.input.VisualTransformation.None,
            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(D.card).border(1.dp, D.hairline, RoundedCornerShape(12.dp)).padding(14.dp).semantics { contentDescription = label })
    }
}

@Composable
fun SettingsScreen(
    s: SessionState, endpoint: String, onConnect: (String, String) -> Unit, onForget: () -> Unit, speak: Boolean, onSpeak: (Boolean) -> Unit,
    reducedPref: Boolean, onReduced: (Boolean) -> Unit, micGranted: Boolean, onRequestMic: () -> Unit, notifyNeedsGrant: Boolean, onRequestNotify: () -> Unit,
    onTestVoice: () -> Unit, onOpenModels: () -> Unit, modelLabel: String,
) {
    var ep by remember { mutableStateOf(endpoint) }
    var token by remember { mutableStateOf("") }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 4.dp), verticalArrangement = Arrangement.spacedBy(24.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Heading("Connection", s.snapshot?.environment)
            Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(D.card).padding(horizontal = 14.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                Dot(when (s.link) { Link.ONLINE -> D.success; Link.CONNECTING -> D.accent; else -> D.amber })
                Text(when (s.link) { Link.ONLINE -> "Connected"; Link.CONNECTING -> "Connecting"; Link.OFFLINE -> "Backend unreachable"; Link.UNAUTHORISED -> "Token rejected"; Link.IDLE -> "Not connected" }, style = T.body, modifier = Modifier.padding(start = 10.dp).weight(1f))
                Text(if (s.hasToken) "token in Keystore" else "no token", style = T.small)
            }
            if (s.error.isNotEmpty()) Text(s.error, style = T.small.copy(color = D.amber), modifier = Modifier.padding(horizontal = 4.dp))
            Field("Endpoint", ep, { ep = it })
            Field(if (s.hasToken) "Replace access token" else "Access token", token, { token = it }, secret = true)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                ActionButton("Connect", { onConnect(ep.trim(), token.trim()); token = "" }, Modifier.weight(1f), enabled = ep.isNotBlank() && (token.isNotBlank() || s.hasToken), bg = D.text, tone = D.page)
                if (s.hasToken) ActionButton("Forget token", onForget, Modifier.weight(1f))
            }
            Text("Over USB, map the phone's loopback to the laptop backend with adb reverse (for example tcp:8765 to the backend's port) and keep http://127.0.0.1:8765. Anywhere else use an HTTPS endpoint; plain HTTP is only allowed to the phone's own loopback.", style = T.small.copy(color = D.text3), modifier = Modifier.padding(horizontal = 4.dp))
        }
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Heading("Model for new tasks")
            Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(D.card).clickable(enabled = s.link == Link.ONLINE) { onOpenModels() }.padding(horizontal = 14.dp, vertical = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(modelLabel, style = T.body, modifier = Modifier.weight(1f))
                androidx.compose.material3.Icon(androidx.compose.ui.res.painterResource(dev.dream.companion.R.drawable.ic_expand), null, tint = D.text2, modifier = Modifier.size(20.dp))
            }
        }
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Heading("Voice")
            Toggle("Speak short task results aloud", speak, onSpeak)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                ActionButton(if (micGranted) "Microphone allowed" else "Allow microphone", onRequestMic, Modifier.weight(1f), dev.dream.companion.R.drawable.ic_mic, enabled = !micGranted)
                ActionButton("Test spoken reply", onTestVoice, Modifier.weight(1f))
            }
            Text("The microphone only listens while you hold, or have tapped, the mic button. There is no background listening.", style = T.small.copy(color = D.text3), modifier = Modifier.padding(horizontal = 4.dp))
        }
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Heading("Display")
            Toggle("Reduce motion (still Aurora and orb)", reducedPref, onReduced)
        }
        if (notifyNeedsGrant) Column(verticalArrangement = Arrangement.spacedBy(10.dp)) { Heading("Notifications"); ActionButton("Allow task notifications", onRequestNotify, Modifier.fillMaxWidth()) }
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Heading("About")
            Text("DREAM runs on the laptop backend. This phone sends goals, watches the camera and controls tasks; tasks keep running on the backend when the app is closed, and pause and cancel always go through the backend scheduler's safe boundaries.", style = T.small, modifier = Modifier.padding(horizontal = 4.dp))
        }
        Spacer(Modifier.height(16.dp))
    }
}

@Composable
fun Toggle(label: String, value: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(D.card).clickable { onChange(!value) }.padding(horizontal = 14.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = T.body, modifier = Modifier.weight(1f))
        Switch(value, onChange, colors = SwitchDefaults.colors(checkedTrackColor = D.accent, checkedThumbColor = D.page, uncheckedTrackColor = D.raised, uncheckedThumbColor = D.text2, uncheckedBorderColor = D.hairline))
    }
}

@Composable
fun ConnectScreen(s: SessionState, endpoint: String, reduced: Boolean, onConnect: (String, String) -> Unit) {
    var ep by remember { mutableStateOf(endpoint) }
    var token by remember { mutableStateOf("") }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Spacer(Modifier.height(60.dp))
        Orb(if (s.link == Link.CONNECTING) dev.dream.companion.data.PresenceState.THINKING else dev.dream.companion.data.PresenceState.OFFLINE, 120.dp, reduced)
        Text("Connect to DREAM", style = T.display)
        Text(if (s.error.isNotEmpty()) s.error else "Enter the backend endpoint and the session access token. The token is encrypted with the Android Keystore and sent only as a request header.", style = T.small)
        Field("DREAM endpoint", ep, { ep = it })
        Field("Access token", token, { token = it }, secret = true)
        Pill(if (s.link == Link.CONNECTING) "Connecting…" else "Connect", { onConnect(ep.trim(), token.trim()) }, modifier = Modifier.fillMaxWidth(), enabled = ep.isNotBlank() && token.isNotBlank(), bg = D.text, tone = D.page)
        Text("USB: adb reverse tcp:8765 tcp:8765, then keep http://127.0.0.1:8765.", style = T.monoSmall)
        Spacer(Modifier.size(24.dp))
    }
}
