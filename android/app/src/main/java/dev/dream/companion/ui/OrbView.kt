package dev.dream.companion.ui

import android.content.Context
import android.graphics.Color as AColor
import android.graphics.PixelFormat
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.view.Choreographer
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
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.sin

/** The same looks as web/app/components/Orb.tsx: colours [ring A, ring B, core], speed, glow, wobble, highlight, trail, awake. */
private data class OrbLook(val colors: List<Int>, val speed: Float, val glow: Float, val wobble: Float, val highlight: Float, val trail: Float, val awake: Float, val opacity: Float)

private fun c(h: String) = AColor.parseColor(h)
private val REST = OrbLook(listOf(c("#648AFF"), c("#5DE4F4"), c("#0E1430")), 0.45f, 1.0f, 0.8f, 0.8f, 1.0f, 0f, 1f)
private fun orbLook(s: PresenceState): OrbLook = when (s) {
    PresenceState.IDLE -> REST
    PresenceState.WAITING -> REST.copy(speed = 0.25f)
    PresenceState.PAUSED -> REST.copy(speed = 0.15f)
    PresenceState.LISTENING -> OrbLook(listOf(c("#5DE4F4"), c("#E6F2FF"), c("#10203A")), 0.9f, 1.5f, 1.3f, 1.2f, 1.2f, 1f, 1f)
    PresenceState.THINKING -> OrbLook(listOf(c("#986BFF"), c("#648AFF"), c("#140F3A")), 1.25f, 1.2f, 1.0f, 1.1f, 1.6f, 0f, 1f)
    PresenceState.VERIFYING -> OrbLook(listOf(c("#986BFF"), c("#648AFF"), c("#140F3A")), 0.45f, 1.6f, 0.5f, 1.1f, 1.6f, 0f, 1f)
    PresenceState.NAVIGATING, PresenceState.INTERACTING -> OrbLook(listOf(c("#648AFF"), c("#986BFF"), c("#10123A")), 1.8f, 1.3f, 1.6f, 1.3f, 1.8f, 1f, 1f)
    PresenceState.OBSERVING -> OrbLook(listOf(c("#5DE4F4"), c("#648AFF"), c("#0B2030")), 0.7f, 1.4f, 0.6f, 1.5f, 1.0f, 0f, 1f)
    PresenceState.RECOVERING, PresenceState.FAILED -> OrbLook(listOf(c("#E18BDF"), c("#DDB26A"), c("#2A1424")), 0.6f, 1.0f, 1.1f, 0.8f, 1.0f, 0f, 1f)
    PresenceState.SAFETY_STOPPED -> OrbLook(listOf(c("#EE8C80"), c("#E8705F"), c("#2A0E0E")), 0.15f, 1.2f, 0.2f, 0.5f, 0.4f, 0f, 1f)
    PresenceState.COMPLETED -> OrbLook(listOf(c("#8DC9A3"), c("#5DE4F4"), c("#0C2620")), 0.5f, 1.3f, 0.8f, 1.0f, 1.0f, 0f, 1f)
    PresenceState.OFFLINE -> OrbLook(listOf(c("#5D6067"), c("#3A3F48"), c("#101114")), 0.06f, 0.7f, 0.3f, 0.3f, 0.4f, 0f, 0.7f)
}

/** DREAM's presence on Android: the React Bits Orb fragment shader (assets/orb.frag, unmodified) in a translucent GL surface. */
@Composable
fun Orb(state: PresenceState, size: Dp, reduced: Boolean, @Suppress("UNUSED_PARAMETER") level: Float = 0f) {
    val ctx = LocalContext.current
    val gles3 = remember { glesThreeAvailable(ctx) }
    if (!gles3) { OrbFallback(state, size); return }
    AndroidView(factory = { OrbView(it) }, modifier = Modifier.size(size), update = { it.setReducedMotion(reduced); it.setState(state) }, onRelease = { it.onPause() })
}

@Composable
private fun OrbFallback(state: PresenceState, size: Dp) {
    androidx.compose.foundation.Canvas(Modifier.size(size)) {
        val r = this.size.minDimension * 0.32f
        drawCircle(androidx.compose.ui.graphics.Brush.radialGradient(listOf(androidx.compose.ui.graphics.Color(0xFFE6EDF5), tintFor(state), androidx.compose.ui.graphics.Color(0xFF1F2A38)), center, r), r)
    }
}

/**
 * The orb's GL runs on its own thread into a TextureView, so it composites like any other view: it clips to rounded
 * cards, stays translucent over the Aurora and fades with its parent (a SurfaceView punches through clipped cards on
 * this device). ~30 fps while visible, one still frame when motion is reduced, nothing while paused or detached.
 */
class OrbView(context: Context) : android.view.TextureView(context), android.view.TextureView.SurfaceTextureListener {
    private val renderer = OrbRenderer(context)
    @Volatile private var reduced = false
    @Volatile private var paused = false
    private var thread: RenderThread? = null

    init { isOpaque = false; surfaceTextureListener = this }

    fun setState(state: PresenceState) { renderer.setTarget(state); thread?.wake() }
    fun setReducedMotion(value: Boolean) { reduced = value; renderer.reduced = value; thread?.wake() }
    fun onPause() { paused = true }
    fun onResume() { paused = false; thread?.wake() }
    override fun onWindowVisibilityChanged(visibility: Int) { super.onWindowVisibilityChanged(visibility); paused = visibility != VISIBLE; thread?.wake() }

    override fun onSurfaceTextureAvailable(st: android.graphics.SurfaceTexture, width: Int, height: Int) {
        thread = RenderThread(st, width, height).also { it.start() }
    }
    override fun onSurfaceTextureSizeChanged(st: android.graphics.SurfaceTexture, width: Int, height: Int) { thread?.resize(width, height) }
    override fun onSurfaceTextureDestroyed(st: android.graphics.SurfaceTexture): Boolean { thread?.finish(); thread = null; return true }
    override fun onSurfaceTextureUpdated(st: android.graphics.SurfaceTexture) {}

    private inner class RenderThread(private val st: android.graphics.SurfaceTexture, @Volatile var w: Int, @Volatile var h: Int) : Thread("dream-orb") {
        private val lock = Object()
        @Volatile private var running = true
        @Volatile private var sized = true
        fun wake() = synchronized(lock) { lock.notifyAll() }
        fun resize(width: Int, height: Int) { w = width; h = height; sized = true; wake() }
        fun finish() { running = false; wake(); join(500) }

        override fun run() {
            val egl = android.opengl.EGL14.eglGetDisplay(android.opengl.EGL14.EGL_DEFAULT_DISPLAY)
            val ver = IntArray(2); android.opengl.EGL14.eglInitialize(egl, ver, 0, ver, 1)
            val attribs = intArrayOf(android.opengl.EGL14.EGL_RED_SIZE, 8, android.opengl.EGL14.EGL_GREEN_SIZE, 8, android.opengl.EGL14.EGL_BLUE_SIZE, 8, android.opengl.EGL14.EGL_ALPHA_SIZE, 8,
                android.opengl.EGL14.EGL_RENDERABLE_TYPE, 0x40 /* EGL_OPENGL_ES3_BIT_KHR */, android.opengl.EGL14.EGL_NONE)
            val configs = arrayOfNulls<android.opengl.EGLConfig>(1); val n = IntArray(1)
            android.opengl.EGL14.eglChooseConfig(egl, attribs, 0, configs, 0, 1, n, 0)
            val ctx = android.opengl.EGL14.eglCreateContext(egl, configs[0], android.opengl.EGL14.EGL_NO_CONTEXT, intArrayOf(android.opengl.EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, android.opengl.EGL14.EGL_NONE), 0)
            val surface = android.opengl.EGL14.eglCreateWindowSurface(egl, configs[0], st, intArrayOf(android.opengl.EGL14.EGL_NONE), 0)
            android.opengl.EGL14.eglMakeCurrent(egl, surface, surface, ctx)
            try {
                renderer.onSurfaceCreated(null, null)
                while (running) {
                    if (sized) { renderer.onSurfaceChanged(null, w, h); sized = false }
                    renderer.onDrawFrame(null)
                    android.opengl.EGL14.eglSwapBuffers(egl, surface)
                    synchronized(lock) { if (running) lock.wait(if (reduced || paused || !isShown) 0L else 33L) }
                }
            } catch (_: InterruptedException) {
            } finally {
                android.opengl.EGL14.eglMakeCurrent(egl, android.opengl.EGL14.EGL_NO_SURFACE, android.opengl.EGL14.EGL_NO_SURFACE, android.opengl.EGL14.EGL_NO_CONTEXT)
                android.opengl.EGL14.eglDestroySurface(egl, surface); android.opengl.EGL14.eglDestroyContext(egl, ctx); android.opengl.EGL14.eglTerminate(egl)
            }
        }
    }
}

/** Mirrors the web component's animation state (time, swirl, orbit, echo, hover, reveal); pointer reach stays at rest. */
private class OrbRenderer(private val context: Context) : GLSurfaceView.Renderer {
    @Volatile var reduced = false
    @Volatile private var target = orbLook(PresenceState.OFFLINE)
    private var shown = arr(target)
    private var previous: PresenceState? = null
    @Volatile private var flashUntil = 0L
    private var program = 0
    private var vao = 0
    private val u = HashMap<String, Int>()
    private var w = 1
    private var h = 1
    private var last = 0L
    private var time = 0f
    private var swirl = 0f
    private var orbit = 0f
    private var echo = 1f
    private var hover = 0f
    private var reveal = 0f

    fun setTarget(s: PresenceState) {
        target = orbLook(s)
        if (s == PresenceState.COMPLETED && previous != null && previous != PresenceState.COMPLETED) { flashUntil = System.nanoTime() + 1_400_000_000L; echo = 0f }
        previous = s
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        val vertex = "#version 300 es\nin vec2 aPosition;\nvoid main(){ gl_Position = vec4(aPosition, 0.0, 1.0); }"
        program = linkProgram(vertex, context.assets.open("orb.frag").bufferedReader().readText())
        val count = IntArray(1); GLES30.glGetProgramiv(program, GLES30.GL_ACTIVE_UNIFORMS, count, 0)
        for (i in 0 until count[0]) { val n = GLES30.glGetActiveUniform(program, i, IntArray(1), 0, IntArray(1), 0); u[n] = GLES30.glGetUniformLocation(program, n) }
        val ids = IntArray(2); GLES30.glGenVertexArrays(1, ids, 0); GLES30.glGenBuffers(1, ids, 1); vao = ids[0]
        GLES30.glBindVertexArray(vao); GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, ids[1])
        val tri = floatArrayOf(-1f, -1f, 3f, -1f, -1f, 3f)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, 24, ByteBuffer.allocateDirect(24).order(ByteOrder.nativeOrder()).asFloatBuffer().put(tri).apply { position(0) }, GLES30.GL_STATIC_DRAW)
        GLES30.glEnableVertexAttribArray(0); GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, 0)
        last = 0L
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) { w = width; h = height; GLES30.glViewport(0, 0, width, height) }

    override fun onDrawFrame(gl: GL10?) {
        val now = System.nanoTime()
        val dt = if (last == 0L) 1f / 30f else ((now - last) / 1e9f).coerceIn(0.0005f, 0.05f)
        last = now
        val goal = arr(target)
        val k = if (reduced) 1f else 1f - exp(-dt / 0.22f)
        for (i in shown.indices) shown[i] += (goal[i] - shown[i]) * k
        val flash = now < flashUntil
        val motion = if (reduced) 0f else 1f
        hover += ((if (shown[6] > 0.5f || flash) 1f else 0f) - hover) * (1f - exp(-dt / 0.16f))
        val step = dt * shown[0] * motion
        time += step; swirl += step * 2f; orbit -= step; orbit = atan2(sin(orbit), cos(orbit))
        echo = min(1f, echo + dt / 1.2f)
        reveal = if (reduced) 1f else min(1f, reveal + dt / 1.1f)
        val back = reveal - 1f
        val grow = 1f + 2.2f * back * back * back + 1.2f * back * back
        val rv = min(1f, reveal / 0.55f).let { it * it * (3 - 2 * it) }

        GLES30.glClearColor(0f, 0f, 0f, 0f); GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glUseProgram(program); GLES30.glBindVertexArray(vao)
        fun f(n: String, v: Float) = GLES30.glUniform1f(u[n] ?: -1, v)
        GLES30.glUniform2f(u["uResolution"] ?: -1, w.toFloat(), h.toFloat())
        f("uTime", time); f("uRotation", 0f); GLES30.glUniform2f(u["uReach"] ?: -1, 0f, 0f); f("uNear", 0f)
        f("uAwake", hover * 0.8f * 0.18f)
        GLES30.glUniform3f(u["uBackground"] ?: -1, 0.031f, 0.039f, 0.063f) // #080A10
        GLES30.glUniform3f(u["uColor1"] ?: -1, shown[7], shown[8], shown[9])
        GLES30.glUniform3f(u["uColor2"] ?: -1, shown[10], shown[11], shown[12])
        GLES30.glUniform3f(u["uColor3"] ?: -1, shown[13], shown[14], shown[15])
        f("uScale", 1f / (0.62f + 0.38f * grow))
        f("uWobble", shown[2]); f("uGlow", shown[1] + if (flash) 0.8f else 0f); f("uBleed", 1f); f("uHighlight", shown[3])
        GLES30.glUniform2f(u["uEcho"] ?: -1, echo * 0.55f, if (echo < 1f) (1f - echo) * (1f - echo) else 0f)
        f("uSwirl", swirl); f("uOrbit", orbit); f("uReveal", rv); f("uOpacity", shown[5]); f("uTrail", shown[4])
        GLES30.glUniform2f(u["uTilt"] ?: -1, 0f, 0f)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
    }

    companion object {
        /** [speed, glow, wobble, highlight, trail, opacity, awake, c1rgb, c2rgb, c3rgb] */
        fun arr(l: OrbLook) = floatArrayOf(l.speed, l.glow, l.wobble, l.highlight, l.trail, l.opacity, l.awake) +
            l.colors.flatMap { listOf(AColor.red(it) / 255f, AColor.green(it) / 255f, AColor.blue(it) / 255f) }
    }
}

internal fun linkProgram(vs: String, fs: String): Int {
    fun compile(type: Int, src: String): Int {
        val s = GLES30.glCreateShader(type); GLES30.glShaderSource(s, src); GLES30.glCompileShader(s)
        val ok = IntArray(1); GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) error("DREAM shader: ${GLES30.glGetShaderInfoLog(s)}")
        return s
    }
    val p = GLES30.glCreateProgram()
    GLES30.glAttachShader(p, compile(GLES30.GL_VERTEX_SHADER, vs)); GLES30.glAttachShader(p, compile(GLES30.GL_FRAGMENT_SHADER, fs))
    GLES30.glBindAttribLocation(p, 0, "aPosition"); GLES30.glLinkProgram(p)
    return p
}
