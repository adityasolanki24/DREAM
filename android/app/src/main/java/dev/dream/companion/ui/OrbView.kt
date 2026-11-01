package dev.dream.companion.ui

import android.content.Context
import android.graphics.Color as AColor
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.GLES30
import android.view.TextureView
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.viewinterop.AndroidView
import dev.dream.companion.data.PresenceState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

/**
 * DREAM's presence on Android: the React Bits Strands fragment shader (assets/strands.frag, unmodified) — the same
 * component the web uses (web/app/components/Strands.tsx), with the same state looks. Pointer reactions are off; it only
 * reflects real presence states.
 */
private data class StrandsLook(val a: Int, val b: Int, val energy: Float, val speed: Float, val listening: Boolean, val opacity: Float)

private fun c(h: String) = AColor.parseColor(h)
private val REST = StrandsLook(c("#648AFF"), c("#5DE4F4"), 0.25f, 0.45f, false, 1f)
private fun strandsLook(s: PresenceState): StrandsLook = when (s) {
    PresenceState.IDLE -> REST
    PresenceState.WAITING -> REST.copy(speed = 0.25f)
    PresenceState.PAUSED -> REST.copy(speed = 0.15f)
    PresenceState.LISTENING -> StrandsLook(c("#5DE4F4"), c("#E6F2FF"), 0.8f, 0.9f, true, 1f)
    PresenceState.THINKING -> StrandsLook(c("#986BFF"), c("#648AFF"), 0.5f, 1.25f, false, 1f)
    PresenceState.VERIFYING -> StrandsLook(c("#986BFF"), c("#648AFF"), 0.3f, 0.45f, false, 1f)
    PresenceState.NAVIGATING, PresenceState.INTERACTING -> StrandsLook(c("#648AFF"), c("#986BFF"), 0.95f, 1.8f, false, 1f)
    PresenceState.OBSERVING -> StrandsLook(c("#5DE4F4"), c("#648AFF"), 0.35f, 0.7f, false, 1f)
    PresenceState.RECOVERING, PresenceState.FAILED -> StrandsLook(c("#E18BDF"), c("#DDB26A"), 0.55f, 0.6f, false, 1f)
    PresenceState.SAFETY_STOPPED -> StrandsLook(c("#EE8C80"), c("#E8705F"), 0.1f, 0.15f, false, 1f)
    PresenceState.COMPLETED -> StrandsLook(c("#8DC9A3"), c("#5DE4F4"), 0.4f, 0.5f, false, 1f)
    PresenceState.OFFLINE -> StrandsLook(c("#7E838F"), c("#4A5058"), 0.05f, 0.06f, false, 0.85f)
}

@Composable
fun Orb(state: PresenceState, size: Dp, reduced: Boolean, level: Float = 0f) {
    val ctx = LocalContext.current
    val gles3 = remember { glesThreeAvailable(ctx) }
    if (!gles3) { PresenceFallback(state, size); return }
    AndroidView(factory = { StrandsView(it) }, modifier = Modifier.size(size), update = { it.setReducedMotion(reduced); it.setState(state, level) }, onRelease = { it.onPause() })
}

@Composable
private fun PresenceFallback(state: PresenceState, size: Dp) {
    androidx.compose.foundation.Canvas(Modifier.size(size)) {
        val r = this.size.minDimension * 0.32f
        drawCircle(androidx.compose.ui.graphics.Brush.radialGradient(listOf(androidx.compose.ui.graphics.Color(0xFFE6EDF5), tintFor(state), androidx.compose.ui.graphics.Color(0xFF1F2A38)), center, r), r)
    }
}

/**
 * Shader host on its own EGL thread drawing into a TextureView, so it composites like any view: it clips to rounded
 * cards, stays translucent over the Aurora and fades with its parent (a SurfaceView punches through clipped cards on
 * this device). ~30 fps while visible, a single still frame when motion is reduced, nothing while paused or hidden.
 */
class StrandsView(context: Context) : TextureView(context), TextureView.SurfaceTextureListener {
    private val renderer = StrandsRenderer(context)
    @Volatile private var paused = false
    private var thread: RenderThread? = null

    init { isOpaque = false; surfaceTextureListener = this }

    fun setState(state: PresenceState, level: Float) { renderer.setTarget(state); renderer.level = level; thread?.wake() }
    fun setReducedMotion(value: Boolean) { renderer.reduced = value; thread?.wake() }
    fun onPause() { paused = true }
    override fun onWindowVisibilityChanged(visibility: Int) { super.onWindowVisibilityChanged(visibility); paused = visibility != VISIBLE; thread?.wake() }

    override fun onSurfaceTextureAvailable(st: SurfaceTexture, width: Int, height: Int) { thread = RenderThread(st, width, height).also { it.start() } }
    override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, width: Int, height: Int) { thread?.resize(width, height) }
    override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean { thread?.finish(); thread = null; return true }
    override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}

    private inner class RenderThread(private val st: SurfaceTexture, @Volatile var w: Int, @Volatile var h: Int) : Thread("dream-strands") {
        private val lock = Object()
        @Volatile private var running = true
        @Volatile private var sized = true
        fun wake() = synchronized(lock) { lock.notifyAll() }
        fun resize(width: Int, height: Int) { w = width; h = height; sized = true; wake() }
        fun finish() { running = false; wake(); join(500) }

        override fun run() {
            val egl = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
            val ver = IntArray(2); EGL14.eglInitialize(egl, ver, 0, ver, 1)
            val attribs = intArrayOf(EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8, EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, 0x40 /* EGL_OPENGL_ES3_BIT_KHR */, EGL14.EGL_NONE)
            val configs = arrayOfNulls<EGLConfig>(1); val n = IntArray(1)
            EGL14.eglChooseConfig(egl, attribs, 0, configs, 0, 1, n, 0)
            val ctx = EGL14.eglCreateContext(egl, configs[0], EGL14.EGL_NO_CONTEXT, intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE), 0)
            val surface = EGL14.eglCreateWindowSurface(egl, configs[0], st, intArrayOf(EGL14.EGL_NONE), 0)
            EGL14.eglMakeCurrent(egl, surface, surface, ctx)
            try {
                renderer.create()
                while (running) {
                    if (sized) { renderer.resize(w, h); sized = false }
                    renderer.draw()
                    EGL14.eglSwapBuffers(egl, surface)
                    synchronized(lock) { if (running) lock.wait(if (renderer.reduced || paused || !isShown) 0L else 33L) }
                }
            } catch (_: InterruptedException) {
            } finally {
                EGL14.eglMakeCurrent(egl, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
                EGL14.eglDestroySurface(egl, surface); EGL14.eglDestroyContext(egl, ctx); EGL14.eglTerminate(egl)
            }
        }
    }
}

/** Mirrors the web component's frame logic (clock, pace, activity, speech envelope, pulse); pointer input stays at rest. */
private class StrandsRenderer(private val context: Context) {
    @Volatile var reduced = false
    @Volatile var level = 0f
    @Volatile private var target = strandsLook(PresenceState.OFFLINE)
    @Volatile private var flashUntil = 0L
    private var previous: PresenceState? = null
    private var shown = arr(target)
    private var program = 0
    private val u = HashMap<String, Int>()
    private var w = 1
    private var h = 1
    private var last = 0L
    private var clock = 0f
    private var pace = 0f
    private var activity = 0f
    private var speechT = 0f
    private var intro = 0f
    private var pulse = 0f
    private var pulseV = 0f

    fun setTarget(s: PresenceState) {
        target = strandsLook(s)
        if (s == PresenceState.COMPLETED && previous != null && previous != PresenceState.COMPLETED) { flashUntil = System.nanoTime() + 1_400_000_000L; pulseV += 0.9f }
        previous = s
    }

    fun create() {
        val vertex = "#version 300 es\nin vec2 position;\nvoid main() { gl_Position = vec4(position, 0.0, 1.0); }\n"
        program = linkProgram(vertex, context.assets.open("strands.frag").bufferedReader().readText(), "position")
        for (name in listOf("uResolution", "uMotion", "uLevel", "uColors", "uCount", "uAmplitude", "uWaviness", "uThickness", "uGlow", "uTaper", "uSpread",
            "uHueShift", "uIntensity", "uSaturation", "uScale", "uGlass", "uRadius", "uRefraction", "uDispersion", "uRim", "uLean", "uIntro", "uPulse", "uLight", "uOpacity"))
            u[name] = GLES30.glGetUniformLocation(program, name)
        val ids = IntArray(2); GLES30.glGenVertexArrays(1, ids, 0); GLES30.glGenBuffers(1, ids, 1)
        GLES30.glBindVertexArray(ids[0]); GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, ids[1])
        val tri = floatArrayOf(-1f, -1f, 3f, -1f, -1f, 3f)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, 24, ByteBuffer.allocateDirect(24).order(ByteOrder.nativeOrder()).asFloatBuffer().put(tri).apply { position(0) }, GLES30.GL_STATIC_DRAW)
        GLES30.glEnableVertexAttribArray(0); GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, 0)
        GLES30.glUseProgram(program)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glClearColor(0f, 0f, 0f, 0f)
        last = 0L
    }

    fun resize(width: Int, height: Int) { w = width; h = height }

    fun draw() {
        val now = System.nanoTime()
        val dt = if (last == 0L) 0f else ((now - last) / 1e9f).coerceIn(0f, 0.05f)
        last = now
        val goal = arr(target)
        val k = if (reduced) 1f else 1f - exp(-dt / 0.22f) // ~600 ms colour/parameter blend, as on the web
        for (i in shown.indices) shown[i] += (goal[i] - shown[i]) * k
        val flash = now < flashUntil
        intro = if (reduced) 1f else min(1f, intro + dt / 1.4f)
        pulseV += (-170f * pulse - 14f * pulseV) * dt
        pulse += pulseV * dt
        val listening = shown[3] > 0.5f
        if (listening && !reduced) speechT += dt
        val voice = if (level > 0f) level.coerceIn(0f, 1f) else if (listening && !reduced) speech(speechT) else 0f
        val energy = min(1f, shown[0] + if (flash) 0.35f else 0f)
        val tgt = min(1.2f, max(energy, voice))
        activity += (tgt - activity) * (1f - exp(-dt / (if (tgt > activity) 0.12f else 0.4f)))
        pace += (min(1f, activity) - pace) * (1f - exp(-dt / 0.8f))
        if (!reduced) clock += dt * shown[1] * (0.6f + 1.6f * pace)
        if (reduced) activity = tgt

        GLES30.glViewport(0, 0, w, h)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        fun f(n: String, v: Float) = GLES30.glUniform1f(u[n] ?: -1, v)
        val sway = 0.75f + 0.45f * pace
        GLES30.glUniform2f(u["uResolution"] ?: -1, w.toFloat(), h.toFloat())
        GLES30.glUniform3f(u["uMotion"] ?: -1,
            (0.26f * sin(0.53f * clock) + 0.1f * sin(1.19f * clock + 1.3f)) * sway,
            1f + (0.06f * sin(0.71f * clock) + 0.035f * sin(1.37f * clock + 0.4f)) * sway,
            0.06f * sin(0.41f * clock + 2f) * sway)
        f("uLevel", activity.coerceIn(0f, 1f))
        GLES30.glUniform3fv(u["uColors"] ?: -1, 8, palette(), 0)
        GLES30.glUniform1i(u["uCount"] ?: -1, 3)
        f("uAmplitude", 1f); f("uWaviness", 1f); f("uThickness", 1f); f("uGlow", 1f); f("uTaper", 1f); f("uSpread", 1f)
        f("uHueShift", 0f); f("uIntensity", 1f); f("uSaturation", 1f); f("uScale", 1f); f("uGlass", 1f); f("uRadius", 0.42f)
        f("uRefraction", 1f); f("uDispersion", 1f); f("uRim", 1f)
        GLES30.glUniform2f(u["uLean"] ?: -1, 0f, 0f)
        f("uIntro", 1f - (1f - intro).pow(3)); f("uPulse", pulse); f("uLight", 0f); f("uOpacity", shown[2])
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
    }

    /** The web host spreads the given colours over 7 shader roles; with two colours that is a straight blend. */
    private fun palette(): FloatArray {
        val out = FloatArray(8 * 3)
        for (i in 0 until 7) {
            val t = i / 6f
            for (k in 0 until 3) out[i * 3 + k] = shown[4 + k] + (shown[7 + k] - shown[4 + k]) * t
        }
        return out
    }

    companion object {
        /** [energy, speed, opacity, listening, a.rgb, b.rgb] */
        fun arr(l: StrandsLook) = floatArrayOf(l.energy, l.speed, l.opacity, if (l.listening) 1f else 0f,
            AColor.red(l.a) / 255f, AColor.green(l.a) / 255f, AColor.blue(l.a) / 255f, AColor.red(l.b) / 255f, AColor.green(l.b) / 255f, AColor.blue(l.b) / 255f)

        private fun smooth(a: Float, b: Float, x: Float): Float { val k = ((x - a) / (b - a)).coerceIn(0f, 1f); return k * k * (3 - 2 * k) }
        /** The component's own synthetic speech envelope, used while listening when no microphone level is supplied. */
        fun speech(t: Float): Float {
            val phrase = smooth(-0.8f, -0.3f, 0.6f * sin(t * 2.7f) + 0.5f * sin(t * 1.53f + 1.2f))
            val syllable = max(0f, sin(t * 17f + 1.6f * sin(t * 5.3f)))
            val stress = 0.6f + 0.4f * sin(t * 3.7f + 0.7f)
            return phrase * (0.35f + 0.65f * syllable) * (0.25f + stress)
        }
    }
}

internal fun linkProgram(vs: String, fs: String, attribute: String = "aPosition"): Int {
    fun compile(type: Int, src: String): Int {
        val s = GLES30.glCreateShader(type); GLES30.glShaderSource(s, src); GLES30.glCompileShader(s)
        val ok = IntArray(1); GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) error("DREAM shader: ${GLES30.glGetShaderInfoLog(s)}")
        return s
    }
    val p = GLES30.glCreateProgram()
    GLES30.glAttachShader(p, compile(GLES30.GL_VERTEX_SHADER, vs)); GLES30.glAttachShader(p, compile(GLES30.GL_FRAGMENT_SHADER, fs))
    GLES30.glBindAttribLocation(p, 0, attribute); GLES30.glLinkProgram(p)
    return p
}
