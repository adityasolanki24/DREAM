package dev.dream.companion.ui

import android.app.ActivityManager
import android.content.Context
import android.graphics.Color as AColor
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.view.Choreographer
import dev.dream.companion.data.PresenceState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.atan2
import kotlin.math.cbrt
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.pow

/** The same looks as web/app/components/aurora/DreamAurora.tsx, keyed to real presence states. */
data class AuroraLook(val speed: Float, val amplitude: Float, val brightness: Float, val glow: Float, val height: Float, val blend: Float, val stops: List<Int>)

private fun c(hex: String) = AColor.parseColor(hex)
val DREAM_PALETTE = listOf(c("#5DE4F4"), c("#648AFF"), c("#986BFF"), c("#E18BDF"))
private val WARNING = listOf(c("#F2A07A"), c("#E8705F"), c("#C2566E"), c("#E8705F"))
private val DIM = listOf(c("#4C6A86"), c("#4F5E8C"), c("#5C5480"), c("#6A5876"))
private val SOFT_FAIL = listOf(c("#8C7BE0"), c("#B07BCB"), c("#D99AB5"), c("#986BFF"))

fun lookFor(state: PresenceState): AuroraLook = when (state) {
    PresenceState.IDLE -> AuroraLook(0.35f, 0.8f, 0.68f, 0.55f, 0.48f, 0.6f, DREAM_PALETTE)
    PresenceState.WAITING -> AuroraLook(0.25f, 0.7f, 0.61f, 0.5f, 0.46f, 0.6f, DREAM_PALETTE)
    PresenceState.PAUSED -> AuroraLook(0.12f, 0.6f, 0.56f, 0.45f, 0.44f, 0.6f, DREAM_PALETTE)
    PresenceState.LISTENING -> AuroraLook(0.5f, 0.9f, 0.92f, 0.7f, 0.54f, 0.55f, DREAM_PALETTE)
    PresenceState.THINKING -> AuroraLook(0.85f, 1.05f, 0.8f, 0.65f, 0.5f, 0.5f, DREAM_PALETTE)
    PresenceState.NAVIGATING -> AuroraLook(1.15f, 1.3f, 0.82f, 0.62f, 0.52f, 0.45f, DREAM_PALETTE)
    PresenceState.INTERACTING -> AuroraLook(1.0f, 1.2f, 0.84f, 0.66f, 0.52f, 0.45f, DREAM_PALETTE)
    PresenceState.OBSERVING -> AuroraLook(0.5f, 0.7f, 0.8f, 0.7f, 0.46f, 0.4f, listOf(c("#5DE4F4"), c("#648AFF"), c("#5DE4F4"), c("#986BFF")))
    PresenceState.VERIFYING -> AuroraLook(0.4f, 0.55f, 0.78f, 0.72f, 0.42f, 0.35f, DREAM_PALETTE)
    PresenceState.COMPLETED -> AuroraLook(0.35f, 0.8f, 0.74f, 0.6f, 0.48f, 0.6f, DREAM_PALETTE)
    PresenceState.RECOVERING -> AuroraLook(0.55f, 0.9f, 0.66f, 0.55f, 0.46f, 0.6f, SOFT_FAIL)
    PresenceState.FAILED -> AuroraLook(0.3f, 0.7f, 0.58f, 0.5f, 0.44f, 0.65f, SOFT_FAIL)
    PresenceState.SAFETY_STOPPED -> AuroraLook(0.18f, 0.5f, 0.72f, 0.6f, 0.46f, 0.5f, WARNING)
    PresenceState.OFFLINE -> AuroraLook(0.08f, 0.5f, 0.36f, 0.4f, 0.4f, 0.7f, DIM)
}

fun glesThreeAvailable(context: Context): Boolean =
    (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).deviceConfigurationInfo.reqGlEsVersion >= 0x30000

/**
 * Native host for the React Bits Aurora fragment shader (assets/aurora.frag, GLSL ES 3.00, unmodified).
 * GLES 3.0 exists on every Android 9 device with a 3.0 GPU (the OPPO A31's PowerVR GE8320 included); RuntimeShader
 * (Android 13) is not used. To keep GPU and battery use low it renders at half resolution, at most ~30 fps, only while
 * attached and resumed, and a single still frame when motion is reduced.
 */
class AuroraView(context: Context) : GLSurfaceView(context) {
    private val renderer = AuroraRenderer(context)
    private var running = false
    private var reduced = false
    private var lastFrame = 0L
    private val tick = object : Choreographer.FrameCallback {
        override fun doFrame(frameTimeNanos: Long) {
            if (!running) return
            if (frameTimeNanos - lastFrame >= 32_000_000L) { lastFrame = frameTimeNanos; requestRender() }
            Choreographer.getInstance().postFrameCallback(this)
        }
    }

    init {
        setEGLContextClientVersion(3)
        setEGLConfigChooser(8, 8, 8, 8, 0, 0)
        setRenderer(renderer)
        renderMode = RENDERMODE_WHEN_DIRTY
        preserveEGLContextOnPause = true
    }

    fun setState(state: PresenceState) { renderer.target = lookFor(state); renderer.pulseIf(state); if (reduced) requestRender() }

    /** Behind a conversation the Aurora steps back so text stays legible (as on the web). */
    fun setDim(value: Boolean) { renderer.dim = if (value) 0.5f else 1f; if (reduced) requestRender() }

    fun setReducedMotion(value: Boolean) { reduced = value; renderer.reduced = value; restart() }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (w > 0 && h > 0) holder.setFixedSize((w / 2).coerceAtLeast(1), (h / 2).coerceAtLeast(1))
    }

    override fun onResume() { super.onResume(); restart() }
    override fun onPause() { running = false; super.onPause() }
    override fun onDetachedFromWindow() { running = false; super.onDetachedFromWindow() }

    private fun restart() {
        val wasRunning = running
        running = !reduced
        if (running && !wasRunning) Choreographer.getInstance().postFrameCallback(tick)
        requestRender()
    }
}

private class AuroraRenderer(private val context: Context) : GLSurfaceView.Renderer {
    @Volatile var target: AuroraLook = lookFor(PresenceState.OFFLINE)
    @Volatile var reduced = false
    @Volatile var dim = 1f
    private var dimShown = 1f
    private var shown: FloatArray = toArray(target)
    private var program = 0
    private var vao = 0
    private val u = HashMap<String, Int>()
    private var width = 1
    private var height = 1
    private var clock = 0f
    private var reveal = 0f
    private var last = 0L
    private var pulseStart = 0L
    private var previous: PresenceState? = null

    fun pulseIf(state: PresenceState) {
        if (state == PresenceState.COMPLETED && previous != PresenceState.COMPLETED && previous != null) pulseStart = System.nanoTime()
        previous = state
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        val vertex = "#version 300 es\nin vec2 aPosition;\nvoid main(){ gl_Position = vec4(aPosition, 0.0, 1.0); }"
        val fragment = context.assets.open("aurora.frag").bufferedReader().readText()
        program = link(vertex, fragment)
        val count = IntArray(1); GLES30.glGetProgramiv(program, GLES30.GL_ACTIVE_UNIFORMS, count, 0)
        for (i in 0 until count[0]) {
            val name = GLES30.glGetActiveUniform(program, i, IntArray(1), 0, IntArray(1), 0).replace("[0]", "")
            u[name] = GLES30.glGetUniformLocation(program, if (name == "uStops") "uStops[0]" else name)
        }
        val ids = IntArray(2); GLES30.glGenVertexArrays(1, ids, 0); GLES30.glGenBuffers(1, ids, 1)
        vao = ids[0]
        GLES30.glBindVertexArray(vao)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, ids[1])
        val tri = floatArrayOf(-1f, -1f, 3f, -1f, -1f, 3f)
        val buf = ByteBuffer.allocateDirect(tri.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(tri).apply { position(0) }
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, tri.size * 4, buf, GLES30.GL_STATIC_DRAW)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, 0)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA) // the shader outputs premultiplied colour
        last = 0L
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) { width = w; height = h; GLES30.glViewport(0, 0, w, h) }

    override fun onDrawFrame(gl: GL10?) {
        val now = System.nanoTime()
        val dt = if (last == 0L) 1f / 30f else ((now - last) / 1e9f).coerceIn(0.0005f, 0.1f)
        last = now
        val goal = toArray(target)
        val k = if (reduced) 1f else 1f - exp(-dt / 0.35f) // ease between states (~0.9 s to settle)
        for (i in shown.indices) shown[i] += (goal[i] - shown[i]) * k
        dimShown += (dim - dimShown) * k
        if (!reduced) { clock += dt * shown[0]; reveal = (reveal + dt / 2.4f).coerceAtMost(1f) } else reveal = 1f
        val p = if (pulseStart == 0L || reduced) 0f else ((now - pulseStart) / 1.8e9f).let { t -> if (t >= 1f) 0f else kotlin.math.sin(Math.PI.toFloat() * (t * 1.4f).coerceAtMost(1f)) * (1f - t) }

        GLES30.glClearColor(0.031f, 0.039f, 0.063f, 1f) // #080A10
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(program)
        GLES30.glBindVertexArray(vao)
        GLES30.glUniform2f(u["uResolution"] ?: -1, width.toFloat(), height.toFloat())
        GLES30.glUniform1f(u["uTime"] ?: -1, clock)
        GLES30.glUniform3fv(u["uStops"] ?: -1, 4, palette(shown), 0)
        GLES30.glUniform1i(u["uStopCount"] ?: -1, 4)
        GLES30.glUniform1f(u["uAmplitude"] ?: -1, shown[1])
        GLES30.glUniform1f(u["uBlend"] ?: -1, shown[5])
        GLES30.glUniform1f(u["uHeight"] ?: -1, (shown[4] + p * 0.05f).coerceIn(0f, 1f))
        GLES30.glUniform1f(u["uScale"] ?: -1, 0.9f)
        GLES30.glUniform1f(u["uGlow"] ?: -1, shown[3] + p * 0.25f)
        GLES30.glUniform1f(u["uBrightness"] ?: -1, (shown[2] * 0.8f + p * 0.45f) * dimShown)
        GLES30.glUniform1i(u["uLayers"] ?: -1, 2)
        GLES30.glUniform1f(u["uFlip"] ?: -1, 0f)
        GLES30.glUniform3f(u["uPointer"] ?: -1, 0f, 0.5f, 0f)
        GLES30.glUniform1f(u["uReveal"] ?: -1, 1f - (1f - reveal).pow(3))
        GLES30.glUniform1f(u["uOpacity"] ?: -1, 1f)
        GLES30.glUniform1f(u["uLightMode"] ?: -1, 0f)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
    }

    private fun link(vs: String, fs: String): Int {
        fun compile(type: Int, src: String): Int {
            val s = GLES30.glCreateShader(type); GLES30.glShaderSource(s, src); GLES30.glCompileShader(s)
            val ok = IntArray(1); GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
            if (ok[0] == 0) error("Aurora shader: ${GLES30.glGetShaderInfoLog(s)}")
            return s
        }
        val p = GLES30.glCreateProgram()
        GLES30.glAttachShader(p, compile(GLES30.GL_VERTEX_SHADER, vs)); GLES30.glAttachShader(p, compile(GLES30.GL_FRAGMENT_SHADER, fs))
        GLES30.glBindAttribLocation(p, 0, "aPosition"); GLES30.glLinkProgram(p)
        return p
    }

    companion object {
        /** [speed, amplitude, brightness, glow, height, blend, r0,g0,b0 … r3,g3,b3] in sRGB 0..1. */
        fun toArray(l: AuroraLook) = floatArrayOf(l.speed, l.amplitude, l.brightness, l.glow, l.height, l.blend) +
            l.stops.flatMap { listOf(AColor.red(it) / 255f, AColor.green(it) / 255f, AColor.blue(it) / 255f) }

        /** Same sRGB → OKLCH conversion and hue unwrapping as the web component. */
        fun palette(a: FloatArray): FloatArray {
            val lch = (0 until 4).map { i -> oklch(a[6 + i * 3], a[7 + i * 3], a[8 + i * 3]) }
            val out = FloatArray(12)
            var prev = lch.firstOrNull { it[1] > 0.02f }?.get(2) ?: 0f
            lch.forEachIndexed { i, s ->
                var h = if (s[1] > 0.02f) s[2] else prev
                while (h - prev > Math.PI) h -= (2 * Math.PI).toFloat()
                while (h - prev < -Math.PI) h += (2 * Math.PI).toFloat()
                prev = h
                out[i * 3] = s[0]; out[i * 3 + 1] = s[1]; out[i * 3 + 2] = h
            }
            return out
        }

        private fun oklch(r0: Float, g0: Float, b0: Float): FloatArray {
            fun lin(c: Float) = if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).pow(2.4f)
            val r = lin(r0); val g = lin(g0); val b = lin(b0)
            val l = cbrt(0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b)
            val m = cbrt(0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b)
            val s = cbrt(0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b)
            val L = 0.2104542553f * l + 0.793617785f * m - 0.0040720468f * s
            val A = 1.9779984951f * l - 2.428592205f * m + 0.4505937099f * s
            val B = 0.0259040371f * l + 0.7827717662f * m - 0.808675766f * s
            return floatArrayOf(L, hypot(A, B), atan2(B, A))
        }
    }
}
