package dev.dream.companion.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import dev.dream.companion.data.DreamApi
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okio.BufferedSource
import java.util.concurrent.TimeUnit

enum class CameraStatus { CONNECTING, LIVE, PAUSED, STALE, NO_FRAMES, UNAVAILABLE, ERROR }

data class CameraState(
    val frame: Bitmap? = null,
    val status: CameraStatus = CameraStatus.CONNECTING,
    val transport: String = "",
    val fps: Float = 0f,
    /** Seconds since the shown frame was captured, from the backend's own capture clock (camera/meta). */
    val ageS: Double? = null,
    val dropped: Int = 0,
    val message: String = "",
)

/**
 * The robot head camera as the backend already serves it: the authenticated MJPEG push stream (/camera/stream, binary
 * JPEG frames, no base64), falling back to conditional single-frame polling (/camera/frame?after=, 204 when unchanged).
 * The phone never asks the simulator to step; when no new frame arrives while the session is online the stage reports
 * "simulation paused or not sensing", not a network failure. Runs only while a camera surface is on screen.
 */
class CameraStream(private val api: DreamApi, private val scope: CoroutineScope) {
    private val _state = MutableStateFlow(CameraState())
    val state: StateFlow<CameraState> = _state
    private var job: Job? = null
    private var metaJob: Job? = null
    private val times = ArrayDeque<Long>()
    @Volatile private var lastFrameAt = 0L

    fun start() {
        if (job?.isActive == true) return
        job = scope.launch(Dispatchers.IO) {
            var failures = 0
            while (isActive) {
                val ok = runCatching { mjpeg() }.isSuccess
                if (!isActive) break
                failures = if (ok) 0 else failures + 1
                // The stream allows two clients; on refusal or error use single-frame polling for a while, then retry.
                runCatching { poll(if (failures > 2) 30_000 else 8_000) }
                delay(1_000)
            }
        }
        metaJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                runCatching {
                    val m = api.cameraMeta()
                    val age = if (m.isNull("frame_age_ms")) null else m.optDouble("frame_age_ms") / 1000.0
                    val noFrames = m.isNull("frame_id")
                    _state.value = _state.value.copy(ageS = age, dropped = m.optInt("dropped_frames"),
                        status = if (noFrames && _state.value.frame == null) CameraStatus.NO_FRAMES else _state.value.status)
                }.onFailure { if (_state.value.frame == null) _state.value = _state.value.copy(status = CameraStatus.UNAVAILABLE, message = "Camera endpoint unavailable") }
                updateFreshness()
                delay(1_500)
            }
        }
    }

    fun stop() { job?.cancel(); metaJob?.cancel(); job = null; metaJob = null }

    private fun accept(bytes: ByteArray, transport: String) {
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.RGB_565 }
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts) ?: return
        val now = System.currentTimeMillis()
        lastFrameAt = now
        times.addLast(now); while (times.isNotEmpty() && now - times.first() > 2_000) times.removeFirst()
        _state.value = _state.value.copy(frame = bmp, status = CameraStatus.LIVE, transport = transport, fps = times.size / 2f, message = "")
    }

    private fun updateFreshness() {
        val s = _state.value
        if (s.frame == null) return
        val idle = System.currentTimeMillis() - lastFrameAt
        val age = s.ageS ?: 0.0
        _state.value = when {
            idle > 1_500 && age > 30 -> s.copy(status = CameraStatus.STALE, fps = 0f)
            idle > 1_500 -> s.copy(status = CameraStatus.PAUSED, fps = 0f)
            else -> s
        }
    }

    private fun mjpeg() {
        val client = api.client.newBuilder().readTimeout(15, TimeUnit.SECONDS).build()
        client.newCall(api.streamRequest()).execute().use { r ->
            if (!r.isSuccessful) error("stream ${r.code}")
            val src = r.body!!.source()
            _state.value = _state.value.copy(transport = "MJPEG stream", status = if (_state.value.frame == null) CameraStatus.CONNECTING else _state.value.status)
            while (true) {
                val length = readPartHeaders(src) ?: return
                accept(src.readByteArray(length.toLong()), "MJPEG stream")
                updateFreshness()
            }
        }
    }

    /** Reads one multipart part's headers; returns its Content-Length, or null at end of stream. */
    private fun readPartHeaders(src: BufferedSource): Int? {
        var length: Int? = null
        var started = false
        while (true) {
            val line = src.readUtf8Line() ?: return null
            if (line.startsWith("--dreamframe")) { started = true; continue }
            if (!started) continue
            if (line.isEmpty()) { if (length != null) return length else continue }
            val i = line.indexOf(':')
            if (i > 0 && line.substring(0, i).equals("Content-Length", true)) length = line.substring(i + 1).trim().toIntOrNull()
        }
    }

    private suspend fun poll(durationMs: Long) {
        val until = System.currentTimeMillis() + durationMs
        var after = ""
        while (System.currentTimeMillis() < until) {
            api.client.newCall(api.frameRequest(after)).execute().use { r ->
                if (r.code == 200) {
                    after = r.header("X-Frame-Id") ?: after
                    accept(r.body!!.bytes(), "JPEG polling")
                } else if (r.code == 401 || r.code == 404) error("frame ${r.code}")
            }
            updateFreshness()
            delay(200)
        }
    }
}
