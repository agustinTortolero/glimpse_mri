// src/main.cpp
#include <QApplication>
#include <QFileInfo>
#include <QDir>
#include <QString>
#include <QDateTime>
#include <QLoggingCategory>
#include <iostream>
#include <sstream>

#include "../view/mainwindow.hpp"
#include "../controller/app_controller.hpp"
#include "logger.hpp"

// -------------------------------------------------------------
// DEV fallback (optional): uncomment to force a local test file
// -------------------------------------------------------------
// #define GLIMPSE_USE_FALLBACK 0
static const char* kFallbackPath =
    "C:\\datasets\\MRI_raw\\FastMRI\\brain_multicoil\\file_brain_AXFLAIR_200_6002452.h5";
// static const char* kFallbackPath =
//     "C:\\datasets\\MRI_DICOM\\IMG-0001-00001_fromSimens.dcm";
// static const char* kFallbackPath =
//     "C:\\datasets\\MRI_raw\\from mridata_dot_org\\52c2fd53-d233-4444-8bfd-7c454240d314.h5";

// Small helper to duplicate a message to console + log
static void log_and_print(simplelog::Logger& log, const std::string& line) {
    std::cerr << line << std::endl;
    log.append(line);
}

// Pick input path: argv[1] (if present). If none, return empty so the UI
// starts idle and waits for drag-and-drop (or menu) instead of auto-loading.
static QString pickInputPath(int argc, char** argv, simplelog::Logger& log) {
#ifdef GLIMPSE_USE_FALLBACK
    (void)argc; (void)argv;
    QString fb = QString::fromUtf8(kFallbackPath);
    log_and_print(log, std::string("[DBG][Main] [DEV] Forcing fallback path: ") + fb.toStdString());
    return fb;
#else
    if (argc > 1 && argv[1] && argv[1][0] != '\0') {
        QString cli = QString::fromLocal8Bit(argv[1]);
        log_and_print(log, std::string("[DBG][Main] Using CLI path: ") + cli.toStdString());
        return cli;
    }
    log_and_print(log, "[DBG][Main] No CLI path. Will NOT auto-load. Waiting for drag-and-drop or menu.");
    return QString(); // empty => no auto-load
#endif
}

// === Qt → Logger bridge ===
static simplelog::Logger* g_logger = nullptr;

static void qt_to_logger_handler(QtMsgType type, const QMessageLogContext& ctx, const QString& msg)
{
    // Console echo (keeps original Qt prefixes)
    QByteArray local = msg.toLocal8Bit();
    switch (type) {
    case QtDebugMsg:    std::cerr << "[QT][DBG] "  << local.constData() << "\n"; break;
    case QtInfoMsg:     std::cerr << "[QT][INFO] " << local.constData() << "\n"; break;
    case QtWarningMsg:  std::cerr << "[QT][WRN] "  << local.constData() << "\n"; break;
    case QtCriticalMsg: std::cerr << "[QT][ERR] "  << local.constData() << "\n"; break;
    case QtFatalMsg:    std::cerr << "[QT][FATAL] "<< local.constData() << "\n"; break;
    }

    // File append
    if (g_logger) {
        std::ostringstream line;
        // Optional: add file:line
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
        if (type == QtFatalMsg) abort();
    }
}

int main(int argc, char** argv)
{
    // === Initialize logger first; it truncates/cleans the file every run ===
    simplelog::Logger log("GlimpseMRI", "app.log");

    // Decorative banner (appears near top of the log)
    simplelog::write_banner(log, {
                                     "Glimpse MRI + MRI Engine V1.1",
                                     "author: Agustin Tortolero",
                                     "contact at: agustin.tortolero@proton.me"
                                 }, '*');

    // Optional run info block
    log.append("----- RUN INFO -----");
#if defined(_DEBUG)
    log.append("Build: Debug");
#else
    log.append("Build: Release");
#endif
#ifdef QT_VERSION_STR
    {
        std::ostringstream qtver;
        qtver << "Qt: " << QT_VERSION_STR << " (runtime " << qVersion() << ")";
        log.append(qtver.str());
    }
#endif
    {
        std::ostringstream when;
        when << "Started at (UTC): " << simplelog::now_utc_iso8601();
        log.append(when.str());
    }
    log.append("--------------------");

    {
        std::ostringstream where;
        where << "[DBG][Main] Log file: " << log.path().string();
        std::cerr << where.str() << "\n";
        log.append(where.str());
    }

    // Install Qt→Logger bridge BEFORE any Qt logs happen
    g_logger = &log;
    qInstallMessageHandler(qt_to_logger_handler);

    try {
        log_and_print(log, "[DBG] Qt app starting.");

        // Some environment/argv context (useful in bug reports)
        {
            std::ostringstream ctx;
            ctx << "[DBG][Main] argc=" << argc;
            for (int i = 0; i < argc; ++i)
                ctx << " argv[" << i << "]=" << (argv[i] ? argv[i] : "(null)");
            log_and_print(log, ctx.str());
        }
#ifdef QT_VERSION_STR
        {
            std::ostringstream qtver;
            qtver << "[DBG][Main] Qt compile-time version: " << QT_VERSION_STR
                  << " | runtime: " << qVersion();
            log_and_print(log, qtver.str());
        }
#endif

        QApplication app(argc, argv);

        MainWindow w;
        AppController controller(&w);   // Qt logs are bridged; controller doesn't need Logger

        // Choose input (CLI or none)
        const QString inPath = pickInputPath(argc, argv, log);
        const QFileInfo fi(inPath);

        if (!inPath.isEmpty()) {
            // Print resolved info (handy for path issues)
            std::ostringstream ss;
            ss << "[DBG][Main] Resolved path:"
               << "\n  absolute = " << fi.absoluteFilePath().toStdString()
               << "\n  exists   = " << (fi.exists() ? "yes" : "NO")
               << "\n  suffix   = " << fi.suffix().toStdString();
            log_and_print(log, ss.str());

            if (!fi.exists()) {
                log_and_print(log, "[WRN][Main] Input file does not exist. Continuing; loaders will report errors.");
            }
        } else {
            log_and_print(log, "[DBG][Main] Starting with no input; user will drag-and-drop or use menu.");
        }

        // Initial title; controller will refine after probe()
        w.setWindowTitle("Glimpse MRI ---- alphaTest");
        log_and_print(log, "[DBG][Main] Window title set.");

        // Load only if we have a path (CLI or dev fallback)
        if (!inPath.isEmpty()) {
            log_and_print(log, "[DBG][Main] Calling controller.load(...)");
            controller.load(inPath);
        } else {
            log_and_print(log, "[DBG][Main] Skipping auto-load; waiting for user action.");
        }

        log_and_print(log, "[DBG][Main] Calling controller.show() (deferred draw)");
        controller.show();

        w.show();
        log_and_print(log, "[DBG][Main] Entering app.exec()");
        int rc = app.exec();
        {
            std::ostringstream ss;
            ss << "[DBG][Main] app.exec() returned rc=" << rc;
            log_and_print(log, ss.str());
        }
        return rc;

    } catch (const std::exception& e) {
        std::ostringstream ss;
        ss << "[FATAL] Unhandled std::exception: " << e.what();
        log_and_print(log, ss.str());
        return 1;
    } catch (...) {
        log_and_print(log, "[FATAL] Unknown exception.");
        return 1;
    }
}
