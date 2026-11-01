package dev.dream.companion.data

import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject

/**
 * Everything the phone needs from a robot, independent of where the robot runs.
 *
 * The app only ever talks to this interface. [DreamApi] implements it over DREAM's authenticated phone API
 * (dream/runtime/phone_api.py): today that is the laptop simulation over USB, tomorrow a robot's on-board DREAM runtime
 * reached over a private network. A robot that serves the same endpoints works unchanged; a different protocol only needs
 * another implementation of this interface, not changes to the UI, the session or the camera.
 */
interface RobotLink {
    val baseUrl: String
    /** HTTP client carrying this link's authentication; reused by the camera stream. */
    val client: OkHttpClient

    suspend fun status(): JSONObject
    suspend fun tasks(): List<Task>
    suspend fun snapshot(): Snapshot
    suspend fun events(after: Int): JSONObject
    suspend fun submit(text: String, choice: ModelChoice): String
    suspend fun act(id: String, action: String): JSONObject
    suspend fun cameraMeta(): JSONObject
    /** What this robot actually offers; probed once per connection so the UI never promises a missing feature. */
    suspend fun capabilities(): Capabilities

    fun streamRequest(): Request
    fun frameRequest(after: String): Request
}

/** Features a connected robot really provides (from its own responses, never assumed). */
data class Capabilities(
    val camera: Boolean = false,
    val webrtc: Boolean = false,
    val homeMemory: Boolean = false,
    val providers: Int = 0,
) {
    companion object { val UNKNOWN = Capabilities() }
}

/** A saved way to reach one robot. The token is stored separately, encrypted by the Android Keystore. */
data class ConnectionProfile(val id: String, val name: String, val endpoint: String) {
    val kind: String get() = EndpointPolicy.kindOf(endpoint)
    fun toJson(): JSONObject = JSONObject().put("id", id).put("name", name).put("endpoint", endpoint)
    companion object {
        fun parse(o: JSONObject) = ConnectionProfile(o.getString("id"), o.optString("name").ifEmpty { "Robot" }, o.getString("endpoint"))
    }
}

/**
 * Which endpoints the app accepts, mirroring res/xml/network_security_config.xml:
 * plain HTTP to the phone's own loopback (USB `adb reverse`) and to Tailscale MagicDNS names (`http://<host>.<tailnet>.ts.net`:
 * those resolve only inside your tailnet and the traffic is WireGuard-encrypted end to end), HTTPS anywhere else (a robot's
 * on-board TLS endpoint or a reverse proxy). Raw IP addresses over a network are refused: robot control is never sent in
 * clear text across a network.
 */
object EndpointPolicy {
    private val loopback = Regex("^http://(127\\.0\\.0\\.1|localhost)(:\\d+)?/?$", RegexOption.IGNORE_CASE)
    private val tailnet = Regex("^http://[a-z0-9-]+(\\.[a-z0-9-]+)*\\.ts\\.net(:\\d+)?/?$", RegexOption.IGNORE_CASE)
    private val https = Regex("^https://[^\\s/:]+(:\\d+)?(/[^\\s]*)?$", RegexOption.IGNORE_CASE)

    /** Null when acceptable, else the reason in plain language. */
    fun check(endpoint: String): String? {
        val e = endpoint.trim()
        return when {
            e.isEmpty() -> "Enter an endpoint."
            loopback.matches(e) || tailnet.matches(e) || https.matches(e) -> null
            e.startsWith("http://", true) -> "Plain http is only allowed over USB (127.0.0.1) or to a tailnet name (…ts.net). Use https:// for other networked robots."
            else -> "Use https://host[:port], http://<name>.ts.net:<port> on your tailnet, or http://127.0.0.1:<port> over USB."
        }
    }

    fun kindOf(endpoint: String): String = when {
        loopback.matches(endpoint.trim()) -> "USB (adb reverse)"
        tailnet.matches(endpoint.trim()) -> "Private tailnet (WireGuard)"
        endpoint.contains(".ts.net", true) -> "Private tailnet (HTTPS)"
        else -> "HTTPS"
    }
}
