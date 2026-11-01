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
 * The same authenticated endpoints the web UI uses. The token travels only in the Authorization header,
 * never in a URL; the base URL is the user's configured DREAM backend (loopback over `adb reverse`, or HTTPS).
 */
class DreamApi(val baseUrl: String, private val token: String) {
    val client: OkHttpClient = shared

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

    suspend fun status() = JSONObject(call("/status"))
    suspend fun tasks(): List<Task> { val a = JSONArray(call("/tasks")); return (0 until a.length()).map { Task.parse(a.getJSONObject(it)) } }
    suspend fun snapshot() = Snapshot.parse(JSONObject(call("/snapshot")))
    suspend fun events(after: Int) = JSONObject(call("/ag-ui/events?after=$after"))
    suspend fun submit(text: String, choice: ModelChoice): String {
        val body = JSONObject().put("text", text).put("priority", 0).put("provider", choice.provider)
        if (choice.model != null && choice.provider != "structured") body.put("model", choice.model)
        return JSONObject(call("/tasks", "POST", body)).getString("id")
    }
    suspend fun act(id: String, action: String) = JSONObject(call("/tasks/${java.net.URLEncoder.encode(id, "UTF-8")}/$action", "POST"))
    suspend fun cameraMeta() = JSONObject(call("/camera/meta"))

    fun streamRequest() = request("/camera/stream").build()
    fun frameRequest(after: String) = request("/camera/frame?after=${java.net.URLEncoder.encode(after, "UTF-8")}").build()

    companion object {
        private val JSON = "application/json".toMediaType()
        val shared: OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(4, TimeUnit.SECONDS).readTimeout(6, TimeUnit.SECONDS).writeTimeout(6, TimeUnit.SECONDS).build()
    }
}
