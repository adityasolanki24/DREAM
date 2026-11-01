package dev.dream.companion.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class ApiError(message: String, val status: Int) : Exception(message)

/**
 * [RobotLink] over DREAM's phone API v1 — the same authenticated endpoints the web UI uses. The token travels only in
 * the Authorization header, never in a URL; the base URL is the selected connection profile (loopback over USB, or HTTPS).
 */
class DreamApi(override val baseUrl: String, private val token: String) : RobotLink {
    override val client: OkHttpClient = shared

    private fun request(path: String) = Request.Builder().url(baseUrl.trimEnd('/') + path).header("Authorization", "Bearer $token")

    private suspend fun call(path: String, method: String = "GET", body: JSONObject? = null): String = withContext(Dispatchers.IO) {
        val rb = request(path).header("Cache-Control", "no-store")
        if (method == "POST") rb.post((body ?: JSONObject()).toString().toRequestBody(JSON))
        try {
            client.newCall(rb.build()).execute().use { r ->
                val text = r.body?.string().orEmpty()
                if (!r.isSuccessful) {
                    val msg = runCatching { JSONObject(text).optString("error") }.getOrNull().orEmpty()
                    throw ApiError(msg.ifEmpty { "Request failed (${r.code})" }, r.code)
                }
                text
            }
        } catch (e: ApiError) { throw e } catch (e: Exception) { throw ApiError("The DREAM backend did not respond.", 0) }
    }

    override suspend fun status() = JSONObject(call("/status"))
    override suspend fun tasks(): List<Task> { val a = JSONArray(call("/tasks")); return (0 until a.length()).map { Task.parse(a.getJSONObject(it)) } }
    override suspend fun snapshot() = Snapshot.parse(JSONObject(call("/snapshot")))
    override suspend fun events(after: Int) = JSONObject(call("/ag-ui/events?after=$after"))
    override suspend fun submit(text: String, choice: ModelChoice): String {
        val body = JSONObject().put("text", text).put("priority", 0).put("provider", choice.provider)
        if (choice.model != null && choice.provider != "structured") body.put("model", choice.model)
        return JSONObject(call("/tasks", "POST", body)).getString("id")
    }
    override suspend fun act(id: String, action: String) = JSONObject(call("/tasks/${java.net.URLEncoder.encode(id, "UTF-8")}/$action", "POST"))
    override suspend fun cameraMeta() = JSONObject(call("/camera/meta"))

    /** Optional endpoints: a 404 means the robot simply does not offer that feature. */
    override suspend fun capabilities(): Capabilities {
        val camera = try { cameraMeta(); true } catch (e: ApiError) { if (e.status == 404) false else throw e }
        val webrtc = runCatching { JSONObject(call("/media/status")).optBoolean("available") }.getOrDefault(false)
        val snap = snapshot()
        return Capabilities(camera = camera, webrtc = webrtc, homeMemory = snap.memoryAvailable, providers = snap.choices.count { it.available })
    }

    override fun streamRequest() = request("/camera/stream").build()
    override fun frameRequest(after: String) = request("/camera/frame?after=${java.net.URLEncoder.encode(after, "UTF-8")}").build()

    companion object {
        private val JSON = "application/json".toMediaType()
        val shared: OkHttpClient = OkHttpClient.Builder()
            // Generous enough for a robot reached over a tailnet or mobile data, short enough to notice a lost link.
            .connectTimeout(8, TimeUnit.SECONDS).readTimeout(10, TimeUnit.SECONDS).writeTimeout(10, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true).build()
    }
}
