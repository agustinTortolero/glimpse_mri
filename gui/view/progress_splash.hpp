#pragma once
#include <QDialog>
#include <QPointer>

class QLabel;
class QProgressBar;
class QTimer;

class ProgressSplash : public QDialog {
    Q_OBJECT
public:
    explicit ProgressSplash(QWidget* owner = nullptr);

    // Start/Show the splash with a base title (pass "Reconstructing")
    void start(const QString& title);

    // Update progress (thread-safe via queued invoke by caller)
    // NOTE: 'stage' is ignored visually (we only show dots), but kept for API stability.
    void updateProgress(int pct, const QString& stage);

    // Hide and schedule deleteLater()
    void finish();

protected:
    bool eventFilter(QObject* obj, QEvent* ev) override;

private:
    void centerOnOwner();
    void tickDots();  // advances the " . . ." animation

    QPointer<QWidget>     m_owner;
    QWidget*              m_card = nullptr;
    QLabel*               m_title = nullptr;  // "Reconstructing"
    QLabel*               m_dots  = nullptr;  // separate line: " . . ."
    QProgressBar*         m_bar   = nullptr;

    // animation state
    QTimer*               m_pulseTimer = nullptr;
    int                   m_pulseStep = 0;    // 0..3
    QString               m_baseTitle;        // e.g., "Reconstructing"
};
