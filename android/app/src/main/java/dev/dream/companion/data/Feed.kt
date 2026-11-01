package dev.dream.companion.data

import org.json.JSONObject

/** Kotlin port of web/app/feed.mjs: folds AG-UI events into one thread per task. */
data class Tool(val id: String, val name: String, val args: JSONObject, val status: String, val message: String)

data class Thread(
    val id: String, val text: String?, val state: String, val tools: List<Tool>,
    val endState: String?, val endReason: String?, val claimSuccess: Boolean?, val claimReason: String?,
)

data class Notice(val tone: String, val title: String, val body: String?, val thread: String?)

data class Feed(val threads: Map<String, Thread> = emptyMap(), val calls: Map<String, String> = emptyMap(), val runs: Map<String, String> = emptyMap())

private fun blank(id: String) = Thread(id, null, "queued", emptyList(), null, null, null, null)

private fun parseJson(text: String): JSONObject = try { JSONObject(text) } catch (_: Exception) { JSONObject() }

fun endTitle(state: String) = when (state) {
    "succeeded" -> "Runtime reports success"
    "failed" -> "Task failed"
    "cancelled" -> "Task cancelled"
    "suspended" -> "Task paused"
    else -> "Task ended"
}

/** Short, human summary of tool arguments. */
fun argSummary(args: JSONObject): String = args.keys().asSequence().joinToString(" · ") { k ->
    when (val v = args.opt(k)) {
        is Double -> "$k ${"%.2f".format(v).trimEnd('0').trimEnd('.')}"
        is Float -> "$k ${"%.2f".format(v).trimEnd('0').trimEnd('.')}"
        else -> "$k $v"
    }
}

fun applyEvents(previous: Feed, events: List<JSONObject>): Pair<Feed, List<Notice>> {
    val threads = previous.threads.toMutableMap()
    val calls = previous.calls.toMutableMap()
    val runs = previous.runs.toMutableMap()
    val notices = mutableListOf<Notice>()
    fun edit(id: String, change: (Thread) -> Thread) { threads[id] = change(threads[id] ?: blank(id)) }
    for (e in events) {
        when (e.optString("type")) {
            "RUN_STARTED" -> {
                val tid = e.optString("threadId"); runs[e.optString("runId")] = tid
                edit(tid) { it.copy(state = "running", endState = null, endReason = null) }
                notices += Notice("info", "Task started", null, tid)
            }
            "TOOL_CALL_START" -> {
                val callId = e.optString("toolCallId"); val tid = callId.replace(Regex("-\\d+$"), ""); calls[callId] = tid
                edit(tid) { it.copy(tools = it.tools + Tool(callId, e.optString("toolCallName"), JSONObject(), "running", "")) }
            }
            "TOOL_CALL_ARGS" -> calls[e.optString("toolCallId")]?.let { tid ->
                val callId = e.optString("toolCallId"); val args = parseJson(e.optString("delta"))
                edit(tid) { t -> t.copy(tools = t.tools.map { if (it.id == callId) it.copy(args = args) else it }) }
            }
            "TOOL_CALL_RESULT" -> calls[e.optString("toolCallId")]?.let { tid ->
                val callId = e.optString("toolCallId"); val result = parseJson(e.optString("content"))
                val status = result.optString("status", "unknown"); val message = result.optString("message", "")
                edit(tid) { t -> t.copy(tools = t.tools.map { if (it.id == callId) it.copy(status = status, message = message) else it }) }
                val name = threads[tid]?.tools?.find { it.id == callId }?.name
                if (result.has("status") && status !in setOf("succeeded", "interrupted")) notices += Notice("warn", "$name $status", message, tid)
            }
            "RUN_FINISHED" -> {
                val r = e.optJSONObject("result"); val state = r?.optString("state") ?: "ended"; val reason = r?.optStringOrNull("reason")
                runs.remove(e.optString("runId")); val tid = e.optString("threadId")
                edit(tid) { it.copy(state = state, endState = state, endReason = reason) }
                notices += Notice(if (state == "succeeded") "good" else "warn", endTitle(state), reason, tid)
            }
            "RUN_ERROR" -> runs.remove(e.optString("runId"))?.let { tid ->
                edit(tid) { it.copy(state = "failed", endState = "failed", endReason = e.optString("message")) }
                notices += Notice("bad", "Task failed", e.optString("message"), tid)
            }
            "CUSTOM" -> {
                val v = e.optJSONObject("value") ?: continue
                val id = v.optString("id").ifEmpty { continue }
                when (v.optString("kind")) {
                    "accepted" -> edit(id) { it.copy(text = v.optString("text")) }
                    "finish" -> edit(id) { it.copy(claimSuccess = v.optBoolean("success"), claimReason = v.optStringOrNull("reason")) }
                    "cancelled" -> if (threads[id]?.endState == null) edit(id) { it.copy(state = "cancelled", endState = "cancelled", endReason = v.optStringOrNull("reason")) }
                }
            }
        }
    }
    return Feed(threads, calls, runs) to notices
}
