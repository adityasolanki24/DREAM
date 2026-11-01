package dev.dream.companion

import android.Manifest
import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner
import dev.dream.companion.audio.Speaker
import dev.dream.companion.data.CredentialStore
import dev.dream.companion.data.Session
import dev.dream.companion.data.SessionEvent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class DreamApp : Application() {
    lateinit var store: CredentialStore
    lateinit var session: Session
    lateinit var speaker: Speaker
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var visible = false

    override fun onCreate() {
        super.onCreate()
        store = CredentialStore(this)
        speaker = Speaker(this)
        session = Session(store, scope, ::onSessionEvent)
        createChannels()
        // Reconnect as soon as a network comes back (Wi-Fi, mobile data or a VPN such as Tailscale) instead of waiting
        // out the offline backoff. registerDefaultNetworkCallback is API 24+, below minSdk 28.
        getSystemService(android.net.ConnectivityManager::class.java).registerDefaultNetworkCallback(object : android.net.ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: android.net.Network) { scope.launch { session.nudge() } }
        })
        ProcessLifecycleOwner.get().lifecycle.addObserver(object : DefaultLifecycleObserver {
            override fun onStart(owner: LifecycleOwner) { visible = true; session.setForeground(true); session.nudge() }
            override fun onStop(owner: LifecycleOwner) { visible = false; session.setForeground(false) }
        })
    }

    private fun onSessionEvent(e: SessionEvent) {
        when (e) {
            is SessionEvent.TaskEnded -> {
                val title = when (e.state) { "succeeded" -> "Runtime reports success"; "failed" -> "Task failed"; "cancelled" -> "Task cancelled"; "suspended" -> "Task paused"; else -> "Task ended" }
                if (e.state == "suspended") return
                if (store.speakReplies && visible) speaker.say("$title. ${e.reason.orEmpty()}")
                if (!visible) notify(CHANNEL_TASKS, e.task.id.hashCode(), "$title · ${e.task.text}", e.reason ?: "")
            }
            // A protective stop is always surfaced, even while the app is open.
            is SessionEvent.SafetyStop -> { notify(CHANNEL_SAFETY, 1, "Robot safety stop", e.reason); if (store.speakReplies && visible) speaker.say("Safety stop. ${e.reason}") }
            is SessionEvent.ConnectionChanged -> if (!visible && !e.online && session.state.value.tasks.any { it.runnable }) notify(CHANNEL_TASKS, 2, "DREAM backend unreachable", "The task continues on the backend if it is still running; reopen DREAM to check.")
            is SessionEvent.ToolProblem -> Unit
        }
    }

    private fun createChannels() {
        // minSdk 28 ≥ 26, so channels always exist.
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL_TASKS, "Task updates", NotificationManager.IMPORTANCE_DEFAULT).apply { description = "Task completion, failure and connection changes" })
        nm.createNotificationChannel(NotificationChannel(CHANNEL_SAFETY, "Robot safety", NotificationManager.IMPORTANCE_HIGH).apply { description = "Protective stops reported by the robot" })
    }

    fun notify(channel: String, id: Int, title: String, body: String) {
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val n = android.app.Notification.Builder(this, channel)
            .setSmallIcon(R.drawable.ic_notify).setContentTitle(title).setContentText(body)
            .setStyle(android.app.Notification.BigTextStyle().bigText(body)).setContentIntent(open).setAutoCancel(true).build()
        getSystemService(NotificationManager::class.java).notify(id, n)
    }

    companion object {
        const val CHANNEL_TASKS = "tasks"
        const val CHANNEL_SAFETY = "safety"
    }
}
