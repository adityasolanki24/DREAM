package dev.dream.companion

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import dev.dream.companion.audio.MicState
import dev.dream.companion.audio.Voice
import dev.dream.companion.camera.CameraState
import dev.dream.companion.camera.CameraStream
import dev.dream.companion.data.Link
import dev.dream.companion.data.presenceFor
import dev.dream.companion.ui.AuroraView
import dev.dream.companion.ui.CompanionScreen
import dev.dream.companion.ui.ComposerHooks
import dev.dream.companion.ui.ConnectScreen
import dev.dream.companion.ui.D
import dev.dream.companion.ui.DreamTheme
import dev.dream.companion.ui.Dot
import dev.dream.companion.ui.FullscreenCamera
import dev.dream.companion.ui.LiveScreen
import dev.dream.companion.ui.MissionScreen
import dev.dream.companion.ui.ModelSheet
import dev.dream.companion.ui.SettingsScreen
import dev.dream.companion.ui.T
import dev.dream.companion.ui.WorldScreen
import dev.dream.companion.ui.glesThreeAvailable
import dev.dream.companion.ui.systemReducedMotion
import kotlinx.coroutines.launch

enum class Tab(val label: String) { COMPANION("Companion"), LIVE("Live"), MISSION("Mission"), WORLD("World"), SETTINGS("Settings") }

class MainActivity : ComponentActivity() {
    private lateinit var voice: Voice

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        voice = Voice(this)
        val app = application as DreamApp
        setContent { DreamTheme { DreamRoot(app, voice) } }
    }

    override fun onStop() { voice.cancel(); super.onStop() } // never keep the microphone open behind the user's back

    override fun onDestroy() { voice.cancel(); super.onDestroy() }
}

@Composable
fun DreamRoot(app: DreamApp, voice: Voice) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    val s by app.session.state.collectAsState()
    val mic by voice.state.collectAsState()
    val partial by voice.partial.collectAsState()
    val micError by voice.error.collectAsState()
    val level by voice.level.collectAsState()
    var tab by rememberSaveable { mutableStateOf(Tab.COMPANION) }
    var draft by rememberSaveable { mutableStateOf("") }
    var modelsOpen by remember { mutableStateOf(false) }
    var fullscreen by remember { mutableStateOf(false) }
    var speak by remember { mutableStateOf(app.store.speakReplies) }
    var reducedPref by remember { mutableStateOf(app.store.reducedMotion) }
    val reduced = reducedPref || systemReducedMotion()
    fun granted(p: String) = ctx.checkSelfPermission(p) == PackageManager.PERMISSION_GRANTED
    var micGranted by remember { mutableStateOf(granted(Manifest.permission.RECORD_AUDIO)) }
    var notifyGranted by remember { mutableStateOf(Build.VERSION.SDK_INT < 33 || granted("android.permission.POST_NOTIFICATIONS")) }
    var pendingMic by remember { mutableStateOf(false) }
    val listening = mic == MicState.LISTENING || mic == MicState.STARTING

    val onTranscript: (String) -> Unit = { text -> draft = if (draft.isBlank()) text else "${draft.trim()} $text" } // into the composer, never auto-sent
    val micLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        micGranted = ok
        if (!ok) Toast.makeText(ctx, "DREAM needs microphone access for voice goals. Typing still works.", Toast.LENGTH_LONG).show()
        else if (pendingMic) Toast.makeText(ctx, "Microphone allowed. Hold the mic button to speak.", Toast.LENGTH_SHORT).show()
        pendingMic = false
    }
    val notifyLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { notifyGranted = it }

    val choice = app.session.effectiveChoice(s)
    val modelLabel = when {
        choice == null -> if (s.link == Link.ONLINE) "No provider available" else "Model"
        choice.provider == "structured" -> "Structured parser"
        else -> choice.model ?: s.choices.find { it.id == choice.provider }?.label ?: choice.provider
    }
    val hooks = ComposerHooks(
        draft = draft, setDraft = { draft = it }, send = { voice.clearError(); app.session.submit(it) },
        mic = mic, micPartial = partial, micError = micError, micLevel = level,
        startMic = {
            if (!micGranted) { pendingMic = true; micLauncher.launch(Manifest.permission.RECORD_AUDIO) }
            else { app.speaker.stop(); voice.start(onTranscript) }
        },
        stopMic = { voice.stop() }, openModels = { modelsOpen = true }, modelLabel = modelLabel,
    )
    val onAct: (String, String) -> Unit = { id, action -> scope.launch { app.session.act(id, action)?.let { Toast.makeText(ctx, it, Toast.LENGTH_LONG).show() } } }

    // The camera stream runs only while a camera surface is visible and the app is in the foreground.
    val api = app.session.currentApi()
    val camera = remember(api) { api?.let { CameraStream(it, scope) } }
    val cam by (camera?.state ?: remember { kotlinx.coroutines.flow.MutableStateFlow(CameraState()) }).collectAsState()
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    val wantCamera = (tab == Tab.LIVE || fullscreen) && s.link != Link.IDLE
    DisposableEffect(camera, wantCamera, lifecycle) {
        fun sync() { if (wantCamera && lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) camera?.start() else camera?.stop() }
        val obs = LifecycleEventObserver { _, _ -> sync() }
        lifecycle.addObserver(obs); sync()
        onDispose { lifecycle.removeObserver(obs); camera?.stop() }
    }

    val shown = s.shown
    val presence = presenceFor(shown, shown?.let { s.feed.threads[it.id] }, s.snapshot?.stopReason, s.link, listening, shown?.let { s.endedAt[it.id] })
    val gles3 = remember { glesThreeAvailable(ctx) }

    Box(Modifier.fillMaxSize().background(D.bg)) {
        // Ambient Aurora: only behind Companion and Connect, where it is the identity, never behind the camera.
        val ambient = !s.hasToken || tab == Tab.COMPANION
        if (ambient) {
            if (gles3) {
                // Full-screen surface: the shader itself confines the light to the top band, so there is no seam.
                AndroidView(factory = { AuroraView(it) }, modifier = Modifier.fillMaxSize(),
                    update = { v -> v.setReducedMotion(reduced); v.setDim(s.hasToken && s.tasks.isNotEmpty()); v.setState(presence.state) }, onRelease = { it.onPause() })
            } else {
                Box(Modifier.fillMaxWidth().fillMaxHeight(0.5f).background(Brush.horizontalGradient(listOf(D.cyan.copy(alpha = 0.18f), D.blue.copy(alpha = 0.2f), D.violet.copy(alpha = 0.18f), D.magenta.copy(alpha = 0.15f)))))
            }
            Box(Modifier.fillMaxSize().background(Brush.verticalGradient(0.3f to Color.Transparent, 0.7f to D.bg.copy(alpha = 0.6f), 1f to D.bg)))
        }

        Column(Modifier.fillMaxSize().imePadding()) {
            Row(Modifier.fillMaxWidth().height(52.dp).padding(start = 18.dp, end = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(15.dp).clip(CircleShape).background(Brush.radialGradient(listOf(D.text, D.accent, Color(0xFF2B3A4C)))))
                Text("DREAM", style = T.label.copy(color = D.text, letterSpacing = T.label.letterSpacing), modifier = Modifier.padding(start = 9.dp))
                Spacer(Modifier.weight(1f))
                Dot(when (s.link) { Link.ONLINE -> D.success; Link.CONNECTING -> D.accent; else -> D.amber })
                Text(when (s.link) { Link.ONLINE -> s.snapshot?.environment?.substringBefore(" (") ?: "Connected"; Link.CONNECTING -> "Connecting"; Link.OFFLINE -> "Offline"; Link.UNAUTHORISED -> "Token rejected"; Link.IDLE -> "Not connected" },
                    style = T.label, modifier = Modifier.padding(start = 6.dp, end = 6.dp), maxLines = 1)
                if (s.hasToken) dev.dream.companion.ui.IconCircle(R.drawable.ic_settings, "Settings", { tab = if (tab == Tab.SETTINGS) Tab.COMPANION else Tab.SETTINGS },
                    size = 36.dp, tint = if (tab == Tab.SETTINGS) D.text else D.text2, bg = if (tab == Tab.SETTINGS) D.veilStrong else Color.Transparent)
            }
            if (s.link == Link.OFFLINE && s.everConnected) Text("Connection lost. Showing the last data received; retrying…", style = T.small.copy(color = D.amber), modifier = Modifier.fillMaxWidth().background(D.amber.copy(alpha = 0.08f)).padding(horizontal = 18.dp, vertical = 6.dp))

            Box(Modifier.weight(1f)) {
                if (!s.hasToken) ConnectScreen(s, app.store.endpoint, reduced) { ep, tok -> app.session.connect(ep, tok) }
                else AnimatedContent(tab, transitionSpec = { fadeIn(androidx.compose.animation.core.tween(180)) togetherWith fadeOut(androidx.compose.animation.core.tween(120)) }, label = "tab") { t ->
                    when (t) {
                        Tab.COMPANION -> CompanionScreen(s, hooks, reduced, onAct) { tab = Tab.MISSION }
                        Tab.LIVE -> LiveScreen(s, cam, reduced, onAct, { fullscreen = true }, listening)
                        Tab.MISSION -> MissionScreen(s, onAct)
                        Tab.WORLD -> WorldScreen(s)
                        Tab.SETTINGS -> SettingsScreen(
                            s, app.store.endpoint, { ep, tok -> if (tok.isNotBlank()) app.session.connect(ep, tok) else app.store.token?.let { app.session.connect(ep, it) } },
                            { app.session.forget() }, speak, { speak = it; app.store.speakReplies = it }, reducedPref, { reducedPref = it; app.store.reducedMotion = it },
                            micGranted, { micLauncher.launch(Manifest.permission.RECORD_AUDIO) }, !notifyGranted, { if (Build.VERSION.SDK_INT >= 33) notifyLauncher.launch("android.permission.POST_NOTIFICATIONS") },
                            { if (app.speaker.isReady) app.speaker.say("DREAM voice replies are working.") else Toast.makeText(ctx, "Text-to-speech is still starting or unavailable.", Toast.LENGTH_SHORT).show() },
                            { modelsOpen = true }, modelLabel,
                        )
                    }
                }
            }
            if (s.hasToken) BottomBar(tab) { tab = it }
        }
        if (modelsOpen) { BackHandler { modelsOpen = false }; ModelSheet(s, choice, { app.session.select(it) }, { modelsOpen = false }) }
        if (fullscreen) { BackHandler { fullscreen = false }; FullscreenCamera(cam, s.link == Link.ONLINE) { fullscreen = false } }
    }
}

@Composable
fun BottomBar(tab: Tab, onTab: (Tab) -> Unit) {
    Row(Modifier.fillMaxWidth().background(D.bg).padding(horizontal = 10.dp, vertical = 8.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
        listOf(Tab.COMPANION, Tab.LIVE, Tab.MISSION, Tab.WORLD).forEach { t ->
            val on = tab == t
            Column(Modifier.clip(RoundedCornerShape(14.dp)).background(if (on) D.veilStrong else Color.Transparent).clickable { onTab(t) }.padding(horizontal = 14.dp, vertical = 8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Box(Modifier.size(width = 18.dp, height = 3.dp).clip(RoundedCornerShape(2.dp)).background(if (on) Brush.horizontalGradient(listOf(D.cyan, D.violet)) else Brush.horizontalGradient(listOf(Color.Transparent, Color.Transparent))))
                Spacer(Modifier.height(5.dp))
                Text(t.label, style = T.label.copy(color = if (on) D.text else D.text3))
            }
        }
    }
}
