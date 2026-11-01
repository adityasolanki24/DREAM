package dev.dream.companion.data

import org.json.JSONArray
import org.json.JSONObject

/** Shapes returned by DREAM's authenticated phone API (dream/runtime/phone_api.py). */
data class Task(
    val id: String, val text: String, val state: String, val priority: Int, val created: Double, val updated: Double,
    val result: String?, val steps: Int, val suspensions: Int, val resume: String, val provider: String?, val model: String?,
) {
    val runnable get() = state in RUNNABLE
    companion object {
        val RUNNABLE = setOf("running", "suspended", "queued")
        fun parse(o: JSONObject) = Task(
            id = o.getString("id"), text = o.optString("text"), state = o.optString("state"), priority = o.optInt("priority"),
            created = o.optDouble("created", 0.0), updated = o.optDouble("updated", 0.0), result = o.optStringOrNull("result"),
            steps = o.optInt("steps"), suspensions = o.optInt("suspensions"), resume = o.optString("resume", "auto"),
            provider = o.optJSONObject("meta")?.optStringOrNull("provider"), model = o.optJSONObject("meta")?.optStringOrNull("model"),
        )
    }
}

data class ProviderChoice(
    val id: String, val label: String, val available: Boolean, val reason: String?, val models: List<String>,
    val defaultModel: String?, val supports: List<String>,
) {
    /** The model used when none is chosen: default_model if listed, else the first, else none. */
    val defaultChoice get() = if (models.isEmpty()) null else defaultModel?.takeIf { it in models } ?: models.first()
}

data class ModelChoice(val provider: String, val model: String?)

data class MapData(val cells: List<IntArray>, val resolution: Double, val origin: DoubleArray, val odometry: String)

data class Candidate(val id: Int, val color: String?, val x: Double, val y: Double, val views: Int)

data class HomeObject(val label: String, val mapId: String, val x: Double, val y: Double, val source: String, val lastSeen: Double)

data class Snapshot(
    val provider: String, val providerLabel: String, val choices: List<ProviderChoice>, val environment: String?,
    val robotTime: Double, val odometry: List<Double>, val stopReason: String?, val cameraTime: Double?, val hasSnapshotCamera: Boolean,
    val map: MapData?, val objects: List<Candidate>, val memoryAvailable: Boolean, val memorySource: String?, val memoryError: String?,
    val memory: List<HomeObject>, val graphTaskId: String?, val outcomeState: String?, val outcomeReason: String?,
) {
    companion object {
        fun parse(o: JSONObject): Snapshot {
            val choices = o.optJSONArray("provider_choices").objects().mapNotNull { c ->
                val id = c.optString("id").ifEmpty { return@mapNotNull null }
                ProviderChoice(
                    id, c.optString("label").ifEmpty { id }, c.optBoolean("available", true), c.optStringOrNull("reason"),
                    c.optJSONArray("models").strings(), c.optStringOrNull("default_model"), c.optJSONArray("supports").strings(),
                )
            }
            val map = o.optJSONObject("map")?.let { m ->
                val cells = m.optJSONArray("cells").arrays().map { col -> IntArray(col.length()) { col.optInt(it) } }
                val origin = m.optJSONArray("origin_m")
                MapData(cells, m.optDouble("resolution_m", 0.1), doubleArrayOf(origin?.optDouble(0) ?: 0.0, origin?.optDouble(1) ?: 0.0), m.optString("odometry"))
            }
            val camera = o.optJSONObject("camera")
            val memory = o.optJSONObject("persistent_memory")
            val graph = o.optJSONObject("task_graph")
            val outcome = o.optJSONObject("outcome")
            return Snapshot(
                provider = o.optString("provider"), providerLabel = o.optString("provider_label"), choices = choices,
                environment = o.optStringOrNull("environment"), robotTime = o.optDouble("robot_time_s", 0.0),
                odometry = o.optJSONArray("odometry").doubles(), stopReason = o.optStringOrNull("stop_reason"),
                cameraTime = camera?.optDouble("time_s"), hasSnapshotCamera = camera?.has("data_url") == true,
                map = map,
                objects = o.optJSONArray("objects").objects().map { c ->
                    val p = c.optJSONArray("position")
                    Candidate(c.optInt("id"), c.optStringOrNull("color"), p?.optDouble(0) ?: 0.0, p?.optDouble(1) ?: 0.0, c.optInt("views"))
                },
                memoryAvailable = memory?.optBoolean("available") == true, memorySource = memory?.optStringOrNull("source"),
                memoryError = memory?.optStringOrNull("error"),
                memory = memory?.optJSONArray("objects").objects().map { h ->
                    val p = h.optJSONArray("position")
                    HomeObject(h.optString("label"), h.optString("map_id"), p?.optDouble(0) ?: 0.0, p?.optDouble(1) ?: 0.0, h.optString("observation_source"), h.optDouble("last_seen", 0.0))
                },
                graphTaskId = graph?.optStringOrNull("task_id"),
                outcomeState = outcome?.optStringOrNull("state"), outcomeReason = outcome?.optStringOrNull("reason"),
            )
        }
    }
}

enum class Link { IDLE, CONNECTING, ONLINE, OFFLINE, UNAUTHORISED }

fun JSONObject.optStringOrNull(key: String): String? = if (isNull(key) || !has(key)) null else opt(key)?.let { if (it is String) it else it.toString() }
fun JSONArray?.objects(): List<JSONObject> = if (this == null) emptyList() else (0 until length()).mapNotNull { optJSONObject(it) }
fun JSONArray?.arrays(): List<JSONArray> = if (this == null) emptyList() else (0 until length()).mapNotNull { optJSONArray(it) }
fun JSONArray?.strings(): List<String> = if (this == null) emptyList() else (0 until length()).mapNotNull { optString(it).ifEmpty { null } }
fun JSONArray?.doubles(): List<Double> = if (this == null) emptyList() else (0 until length()).map { optDouble(it) }
