package dev.dream.companion.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import dev.dream.companion.R

/** The desktop dark tokens (web/app/style.css), so phone and laptop read as one product. */
object D {
    val bg = Color(0xFF080A10)
    val page = Color(0xFF0D0E10)
    val card = Color(0xFF141619)
    val raised = Color(0xFF1B1E22)
    val hairline = Color(0xFF24272C)
    val hairlineStrong = Color(0xFF6B7078)
    val text = Color(0xFFECE8E0)
    val text2 = Color(0xFFA19F99)
    val text3 = Color(0xFF5D6067)
    val accent = Color(0xFF93B4D6)
    val amber = Color(0xFFDDB26A)
    val danger = Color(0xFFEE8C80)
    val success = Color(0xFF8DC9A3)
    val veil = Color(0x0AECE8E0)
    val veilStrong = Color(0x14ECE8E0)
    val glass = Color(0xA60A0B0D)
    val cyan = Color(0xFF5DE4F4)
    val blue = Color(0xFF648AFF)
    val violet = Color(0xFF986BFF)
    val magenta = Color(0xFFE18BDF)
}

val Geist = FontFamily(Font(R.font.geist_regular, FontWeight.Normal), Font(R.font.geist_medium, FontWeight.Medium), Font(R.font.geist_medium, FontWeight.SemiBold))
val GeistMono = FontFamily(Font(R.font.geist_mono, FontWeight.Normal))

object T {
    val display = TextStyle(fontFamily = Geist, fontWeight = FontWeight.Medium, fontSize = 30.sp, lineHeight = 34.sp, letterSpacing = (-0.035).em, color = D.text)
    val title = TextStyle(fontFamily = Geist, fontWeight = FontWeight.Medium, fontSize = 19.sp, lineHeight = 24.sp, letterSpacing = (-0.02).em, color = D.text)
    val body = TextStyle(fontFamily = Geist, fontSize = 15.sp, lineHeight = 21.sp, color = D.text)
    val small = TextStyle(fontFamily = Geist, fontSize = 13.sp, lineHeight = 18.sp, color = D.text2)
    val label = TextStyle(fontFamily = Geist, fontWeight = FontWeight.Medium, fontSize = 12.sp, lineHeight = 16.sp, color = D.text2)
    val mono = TextStyle(fontFamily = GeistMono, fontSize = 12.5.sp, lineHeight = 18.sp, color = D.text)
    val monoSmall = TextStyle(fontFamily = GeistMono, fontSize = 11.sp, lineHeight = 15.sp, color = D.text2)
}

@Composable
fun DreamTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = D.text, onPrimary = D.page, secondary = D.accent, background = D.bg, surface = D.card, onSurface = D.text,
            surfaceVariant = D.raised, onSurfaceVariant = D.text2, outline = D.hairline, error = D.danger,
        ),
        typography = Typography(bodyLarge = T.body, bodyMedium = T.small, labelLarge = T.label, titleLarge = T.title),
        content = content,
    )
}
