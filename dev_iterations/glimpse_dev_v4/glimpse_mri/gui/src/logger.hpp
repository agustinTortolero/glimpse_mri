#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <algorithm>  // <-- for std::max
#include <vector>     // <-- for std::vector

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX          // <-- prevent Windows macros min/max
#endif
#include <windows.h>
#include <shlobj_core.h>  // SHGetKnownFolderPath
#include <combaseapi.h>   // CoTaskMemFree
#pragma comment(lib, "Shell32.lib")
#pragma comment(lib, "Ole32.lib")
#endif


namespace simplelog {

inline std::string now_utc_iso8601()
{
    auto tp = std::chrono::system_clock::now();
    std::time_t t  = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

// Choose a per-user base folder that is “normal” and won’t trip AV heuristics.
inline std::filesystem::path pick_user_documents_or_state_folder(const std::string& appName)
{
    std::error_code ec;

#if defined(_WIN32)
    std::wcout << L"[DBG][log] Detecting Windows Documents folder via KNOWNFOLDERID...\n";
    PWSTR pathW = nullptr;
    HRESULT hr = SHGetKnownFolderPath(FOLDERID_Documents, KF_FLAG_DEFAULT, nullptr, &pathW);
    std::filesystem::path base;
    if (SUCCEEDED(hr) && pathW) {
        base = std::filesystem::path(pathW);
        CoTaskMemFree(pathW);
        std::wcout << L"[DBG][log] Documents resolved to: " << base.wstring() << L"\n";
    } else {
        std::wcerr << L"[DBG][log] SHGetKnownFolderPath failed, falling back to USERPROFILE\\Documents...\n";
        wchar_t* userProfile = _wgetenv(L"USERPROFILE");
        if (userProfile) {
            base = std::filesystem::path(userProfile) / "Documents";
        } else {
            // Last-ditch: current dir
            base = std::filesystem::current_path();
        }
    }
    auto p = base / appName / "logs";
    if (!std::filesystem::exists(p, ec)) {
        std::wcout << L"[DBG][log] Creating directory: " << p.wstring() << L"\n";
        std::filesystem::create_directories(p, ec);
        if (ec) std::wcerr << L"[DBG][log][WARN] CreateDirectories error: " << ec.message().c_str() << L"\n";
    }
    return p;
#else
    std::cout << "[DBG][log] Non-Windows: trying ~/Documents/" << appName << "/logs\n";
    const char* home = std::getenv("HOME");
    std::filesystem::path base = (home ? std::filesystem::path(home) : std::filesystem::current_path());

    auto docs = base / "Documents";
    std::filesystem::path p;
    if (std::filesystem::exists(docs, ec)) {
        p = docs / appName / "logs";
    } else {
        // Prefer XDG_STATE_HOME if available, else ~/.local/state
        const char* xdg = std::getenv("XDG_STATE_HOME");
        if (xdg && std::string(xdg).size()) {
            p = std::filesystem::path(xdg) / appName / "logs";
        } else {
            p = base / ".local" / "state" / appName / "logs";
        }
    }
    if (!std::filesystem::exists(p, ec)) {
        std::cout << "[DBG][log] Creating directory: " << p.string() << "\n";
        std::filesystem::create_directories(p, ec);
        if (ec) std::cerr << "[DBG][log][WARN] CreateDirectories error: " << ec.message() << "\n";
    }
    return p;
#endif
}

struct Logger {
    std::filesystem::path logPath;
    std::ofstream stream;

    // Creates folder, truncates file on construction, writes a header.
    explicit Logger(const std::string& appName = "GlimpseMRI",
                    const std::string& fileName = "app.log")
    {
        auto dir = pick_user_documents_or_state_folder(appName);
        logPath = dir / fileName;

        std::error_code ec;
        auto parent = logPath.parent_path();
        if (!std::filesystem::exists(parent, ec)) {
            std::cout << "[DBG][log] Creating parent dir: " << parent.string() << "\n";
            std::filesystem::create_directories(parent, ec);
            if (ec) std::cerr << "[DBG][log][WARN] CreateDirectories error: " << ec.message() << "\n";
        }

        // Truncate each run to ensure a clean file.
        std::cout << "[DBG][log] Opening (truncate) log file: " << logPath.string() << "\n";
        stream.open(logPath, std::ios::out | std::ios::trunc);
        if (!stream.is_open()) {
            std::cerr << "[DBG][log][ERR] Could not open log file for writing.\n";
            return;
        }

        // Optional: verify it's empty after trunc
        auto sz = std::filesystem::exists(logPath, ec) ? std::filesystem::file_size(logPath, ec) : 0;
        std::cout << "[DBG][log] File size after truncate: " << sz << " bytes\n";

        // Write a simple header
        stream << "===== " << appName << " Log Started: " << now_utc_iso8601() << " =====\n";
        stream.flush();
    }

    // Append one line (adds timestamp automatically)
    void append(const std::string& line)
    {
        if (!stream.is_open()) {
            std::cerr << "[DBG][log][ERR] append() called but stream is not open.\n";
            return;
        }
        stream << "[" << now_utc_iso8601() << "] " << line << "\n";
        stream.flush();
    }

    // Get the full path to show users / for debugging
    std::filesystem::path path() const { return logPath; }
};

// --- Pretty banner helpers -------------------------------------------------
inline void write_banner(std::ofstream& os,
                         const std::vector<std::string>& lines,
                         char border = '*')
{
    if (!os.is_open()) return;

    // compute inner width
    std::size_t inner = 0;
    for (const auto& s : lines) inner = std::max(inner, s.size());

    const std::string top(inner + 6, border); // 2 spaces padding + borders
    os << top << "\n";

    // empty padding line
    {
        std::string pad(inner + 2, ' ');
        os << border << " " << pad << " " << border << "\n";
    }

    for (const auto& s : lines) {
        std::string pad(inner - s.size(), ' ');
        os << border << " " << s << pad << " " << border << "\n";
    }

    // empty padding line
    {
        std::string pad(inner + 2, ' ');
        os << border << " " << pad << " " << border << "\n";
    }

    os << top << "\n";
    os.flush();
}

inline void write_banner(Logger& log,
                         const std::vector<std::string>& lines,
                         char border = '*')
{
    if (!log.stream.is_open()) {
        std::cerr << "[DBG][log][WARN] write_banner called but stream closed.\n";
        return;
    }
    write_banner(log.stream, lines, border);
}
// ---------------------------------------------------------------------------


} // namespace simplelog
