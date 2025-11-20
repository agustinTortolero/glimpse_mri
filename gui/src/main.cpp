// gui/src/main.cpp  — Splash with sharp scaling + HiDPI, Option B exact hold
#include <QApplication>
#include <QPixmap>
#include <QSplashScreen>
#include <QScreen>
#include <QGuiApplication>
#include <QCursor>
#include <QString>
#include <QLoggingCategory>
#include <QEventLoop>
#include <QTimer>
#include <QFont>
#include <QIcon>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>

// ---- Your app headers (adjust if paths differ) ----
#include "../view/mainwindow.hpp"
#include "../controller/app_controller.hpp"
#include "../model/engine_api.h"

#include "logger.hpp"

// ======================== Utilities & Qt log bridge =========================
namespace {

// ===== Engine → Qt logger bridge (line-oriented) ============================
static void engineLogToApp(const char* line, void* user) {
    // user is a simplelog::Logger*, pass it when registering
    auto* lg = static_cast<simplelog::Logger*>(user);
    if (!line) return;

    // Mirror to stderr (so it shows in Qt Creator console)
    std::cerr << line << "\n";

    // Append to file logger
    if (lg) lg->append(line);
}


simplelog::Logger* g_logger = nullptr;

void log_and_print(simplelog::Logger& log, const std::string& line) {
    std::cerr << line << std::endl;
    log.append(line);
}

void qt_to_logger_handler(QtMsgType type, const QMessageLogContext& ctx, const QString& msg) {
    const QByteArray local = msg.toLocal8Bit();
    switch (type) {
    case QtDebugMsg:    std::cerr << "[QT][DBG] "  << local.constData() << "\n"; break;
    case QtInfoMsg:     std::cerr << "[QT][INF] "  << local.constData() << "\n"; break;
    case QtWarningMsg:  std::cerr << "[QT][WRN] "  << local.constData() << "\n"; break;
    case QtCriticalMsg: std::cerr << "[QT][ERR] "  << local.constData() << "\n"; break;
    case QtFatalMsg:    std::cerr << "[QT][FATAL] "<< local.constData() << "\n"; break;
    }
    if (!g_logger) return;

    std::ostringstream line;
    if (ctx.file && ctx.line > 0) {
        line << "[QT][" << (ctx.function ? ctx.function : "?") << " @ "
             << ctx.file << ":" << ctx.line << "] ";
    } else {
        line << "[QT] ";
    }
    switch (type) {
    case QtDebugMsg:    line << "DBG "; break;
    case QtInfoMsg:     line << "INF "; break;
    case QtWarningMsg:  line << "WRN "; break;
    case QtCriticalMsg: line << "ERR "; break;
    case QtFatalMsg:    line << "FATAL "; break;
    }
    line << local.constData();
    g_logger->append(line.str());
    if (type == QtFatalMsg) std::abort();
}

simplelog::Logger createLogger() {
    simplelog::Logger log("GlimpseMRI", "app.log");
    simplelog::write_banner(log, {
                                     "Glimpse MRI — startup",
                                     "Author: Agustin Tortolero",
                                     std::string("Started at (UTC): ") + simplelog::now_utc_iso8601()
                                 }, '=');
    std::ostringstream where; where << "[DBG][Main] Log file: " << log.path().string();
    std::cerr << where.str() << "\n"; log.append(where.str());
    return log;
}

void installQtLogBridge(simplelog::Logger& log) {
    g_logger = &log;
    qInstallMessageHandler(qt_to_logger_handler);
    log_and_print(log, "[DBG][Main] Qt→Logger bridge installed.");
}

// =============================== Splash (sharp) =============================
void splashMessage(QSplashScreen* splash, const QString& msg) {
    if (!splash) return;
    splash->showMessage(msg, Qt::AlignHCenter | Qt::AlignBottom, Qt::white);
    QApplication::processEvents();
}

std::unique_ptr<QSplashScreen> createAndShowSplash(simplelog::Logger& log,
                                                   const char* resourcePath,
                                                   double screenFrac,
                                                   int clampMaxW,
                                                   int clampMaxH)
{
    log_and_print(log, std::string("[DBG][Splash] Loading: ") + resourcePath);

    QPixmap orig(resourcePath);
    if (orig.isNull()) {
        log_and_print(log, "[WRN][Splash] NULL pixmap; skipping");
        return {};
    }

    QScreen* scr = QGuiApplication::screenAt(QCursor::pos());
    if (!scr)
        scr = QGuiApplication::primaryScreen();

    const QRect avail = scr ? scr->availableGeometry() : QRect(0, 0, 1280, 720);
    const qreal dpr   = scr ? scr->devicePixelRatio() : 1.0;

    int maxW = int(avail.width()  * screenFrac);
    int maxH = int(avail.height() * screenFrac);
    if (clampMaxW > 0) maxW = std::min(maxW, clampMaxW);
    if (clampMaxH > 0) maxH = std::min(maxH, clampMaxH);
    const QSize logicalTarget(maxW, maxH);

    const QSize deviceTarget(
        int(std::lround(logicalTarget.width()  * dpr)),
        int(std::lround(logicalTarget.height() * dpr))
        );

    QPixmap pix;
    const bool fits = (orig.width() <= deviceTarget.width()) &&
                      (orig.height() <= deviceTarget.height());

    if (fits) {
        pix = orig;
        pix.setDevicePixelRatio(dpr);
        log_and_print(log, "[DBG][Splash] Using native size (no scale), DPR set.");
    } else {
        QImage img    = orig.toImage();
        QImage scaled = img.scaled(deviceTarget,
                                   Qt::KeepAspectRatio,
                                   Qt::SmoothTransformation);
        pix = QPixmap::fromImage(scaled);
        pix.setDevicePixelRatio(dpr);

        std::ostringstream ss;
        ss << "[DBG][Splash] Smooth downscale to device "
           << scaled.width() << "x" << scaled.height()
           << " (logical " << (scaled.width() / dpr) << "x"
           << (scaled.height() / dpr) << ") DPR=" << dpr;
        log_and_print(log, ss.str());
    }

    auto splash = std::make_unique<QSplashScreen>(pix);
    splash->setWindowFlag(Qt::WindowStaysOnTopHint, true);

    {
        QPalette pal = splash->palette();
        pal.setColor(QPalette::Window, Qt::black);
        splash->setAutoFillBackground(true);
        splash->setPalette(pal);
        qDebug() << "[Splash][UI] Background set to black";
    }


    const QSizeF logicalSize = pix.deviceIndependentSize();
    const QPoint center = avail.center() - QPoint(
                              int(logicalSize.width()  / 2.0),
                              int(logicalSize.height() / 2.0)
                              );
    splash->move(center);

    // --- DejaVu Sans for splash messages (e.g., "Starting UI...") ---
    {
        QFont f("DejaVu Sans");                       // requested font
        // Bump size a bit relative to default
        const qreal basePt = splash->font().pointSizeF();
        if (basePt > 0.0)
            f.setPointSizeF(basePt + 1.5);
        splash->setFont(f);
    }

    splash->show();
    splash->raise();
    splash->activateWindow();

    {
        std::ostringstream ss;
        ss << "[DBG][Splash] show at " << splash->x() << "," << splash->y()
           << " logical=" << logicalSize.width() << "x" << logicalSize.height();
        log_and_print(log, ss.str());
    }

    // Initial splash text – also in DejaVu Sans now
    splash->showMessage(
        "Initializing.",
        Qt::AlignHCenter | Qt::AlignBottom,
        Qt::white
        );
    QApplication::processEvents();

    return splash;
}


// ============================ App icon handling =============================
QIcon buildAppIcon()
{
    QIcon ico;
    ico.addFile(":/icons/mri_16.png");
    ico.addFile(":/icons/mri_20.png");
    ico.addFile(":/icons/mri_24.png");
    ico.addFile(":/icons/mri_32.png");
    ico.addFile(":/icons/mri_40.png");
    ico.addFile(":/icons/mri_48.png");
    ico.addFile(":/icons/mri_64.png");
    ico.addFile(":/icons/mri_72.png");
    ico.addFile(":/icons/mri_96.png");
    ico.addFile(":/icons/mri_128.png");
    ico.addFile(":/icons/mri_192.png");
    ico.addFile(":/icons/mri_256.png");
    ico.addFile(":/icons/mri_512.png");

    const auto sizesN = ico.availableSizes(QIcon::Normal, QIcon::Off);
    if (sizesN.isEmpty()) {
        qWarning() << "[WRN][Icon] No available sizes detected. Check :/icons/* in assets.qrc";
    } else {
        QStringList s; s.reserve(sizesN.size());
        for (const QSize& z : sizesN) s << QString("%1x%2").arg(z.width()).arg(z.height());
        qDebug() << "[DBG][Icon] Normal sizes:" << s.join(", ");
    }
    qDebug() << "[DBG][Icon] App icon built from QRC :/icons/*";
    return ico;
}

// =============================== Build UI ==================================
struct Ui { std::unique_ptr<MainWindow> window; std::unique_ptr<AppController> controller; };

Ui buildUi(simplelog::Logger& log, QSplashScreen* splash) {
    splashMessage(splash, "Creating MainWindow...");
    auto w = std::make_unique<MainWindow>();

    splashMessage(splash, "Wiring controller...");
    auto c = std::make_unique<AppController>(w.get());

    w->setWindowTitle("Glimpse MRI --- preRelease v2");
    log_and_print(log, "[DBG][Main] Window title set.");
    return { std::move(w), std::move(c) };
}

void showUiAndFinishSplash(MainWindow& w, AppController& controller, QSplashScreen* splash, simplelog::Logger& log) {
    splashMessage(splash, "Starting UI...");
    controller.show();
    w.show();
    log_and_print(log, "[DBG][Main] MainWindow shown.");

    if (splash) {
        splash->finish(&w);
        log_and_print(log, "[DBG][Splash] splash.finish(&w)");
    }
}

int runEventLoop(QApplication& app, simplelog::Logger& log) {
    log_and_print(log, "[DBG][Main] Entering app.exec()");
    const int rc = app.exec();
    std::ostringstream ss; ss << "[DBG][Main] app.exec() returned rc=" << rc; log_and_print(log, ss.str());
    return rc;
}

} // namespace

// ================================== main ====================================
int main(int argc, char** argv) {
    auto log = createLogger();
    installQtLogBridge(log);
    engine_set_log_cb(&engineLogToApp, &log);
    log_and_print(log, "[DBG][Main] Engine log callback installed.");

    try {
        log_and_print(log, "[DBG] Qt app starting.");
        {
            std::ostringstream ctx;
            ctx << "[DBG][Main] argc=" << argc;
            for (int i = 0; i < argc; ++i) ctx << " argv[" << i << "]=" << (argv[i] ? argv[i] : "(null)");
            log_and_print(log, ctx.str());
        }


        QApplication GlimpseMRI(argc, argv);

        // --- App icon (taskbar / dock / titlebar) ----------------------------
        QApplication::setWindowIcon(buildAppIcon());

#ifdef QT_VERSION_STR
        {
            std::ostringstream qtver;
            qtver << "[DBG][Main] Qt compile-time: " << QT_VERSION_STR
                  << " | runtime: " << qVersion();
            log_and_print(log, qtver.str());
        }
#endif

        // 1) Show crisp splash (on top)
        auto splash = createAndShowSplash(log,
                                          /*resourcePath*/":/assets/splash.png",
                                          /*screenFrac*/ 0.50,
                                          /*clampMaxW*/  900,
                                          /*clampMaxH*/  600);

        // 2) EXACT HOLD before creating UI
        const int exactMs = 5000;
        {
            std::ostringstream ss; ss << "[DBG][Splash] Holding splash for exact " << exactMs << " ms.";
            log_and_print(log, ss.str());
        }
        {
            QEventLoop loop;
            QTimer::singleShot(exactMs, &loop, &QEventLoop::quit);
            loop.exec();
        }

        // 3) Build UI
        Ui ui = buildUi(log, splash.get());

        // 4) Show UI and close splash
        showUiAndFinishSplash(*ui.window, *ui.controller, splash.get(), log);
        splash.reset();

        // 5) Event loop
        return runEventLoop(GlimpseMRI, log);

    } catch (const std::exception& e) {
        std::ostringstream ss; ss << "[FATAL] Unhandled std::exception: " << e.what();
        log_and_print(log, ss.str()); return 1;
    } catch (...) {
        log_and_print(log, "[FATAL] Unknown exception."); return 1;
    }
}
