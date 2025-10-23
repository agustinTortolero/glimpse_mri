#include "progress_splash.hpp"

#include <QVBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QApplication>
#include <QScreen>
#include <QEvent>
#include <QTimer>
#include <QDebug>

ProgressSplash::ProgressSplash(QWidget* owner)
    : QDialog(owner)
    , m_owner(owner)
{
    qDebug() << "[SPLASH][ctor] owner?" << (owner ? "YES" : "NO");

    // Frameless, always on top, modal to the whole app (blocks clicks)
    setWindowFlags(Qt::Tool | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);
    setModal(false);
    setWindowModality(Qt::ApplicationModal);

    // Make the DIALOG draw an opaque black background (no transparency)
    // (Remove WA_TranslucentBackground; enable styled background; set black bg)
    setAttribute(Qt::WA_StyledBackground, true);
    setStyleSheet("background-color: #000000;"); // solid black
    qDebug() << "[SPLASH] dialog background set to black";

    // Build content on a transparent card so the dialog's black shows through
    m_card = new QWidget(this);
    m_card->setObjectName("SplashCard");
    m_card->setStyleSheet(
        "#SplashCard { background: transparent; }"  // let the dialog's black show
        "QLabel { color: white; }"
        "QProgressBar { color: white; }"
        );

    auto* lay = new QVBoxLayout(m_card);
    lay->setContentsMargins(22, 18, 22, 18);
    lay->setSpacing(8);

    m_title = new QLabel("Reconstructing", m_card);
    m_title->setAlignment(Qt::AlignCenter);
    m_title->setStyleSheet("font-size: 18px; font-weight: 600;");

    m_dots = new QLabel("", m_card);              // separate line for dots
    m_dots->setAlignment(Qt::AlignCenter);
    m_dots->setStyleSheet("font-size: 16px; color: #f0f0f0;");

    m_bar = new QProgressBar(m_card);
    m_bar->setRange(0, 100);
    m_bar->setValue(0);
    m_bar->setTextVisible(true);

    lay->addWidget(m_title);
    lay->addWidget(m_dots);
    lay->addWidget(m_bar);

    resize(440, 170);
    m_card->setGeometry(rect());

    if (m_owner) m_owner->installEventFilter(this);

    // Dots animation timer
    m_pulseTimer = new QTimer(this);
    m_pulseTimer->setInterval(350); // ms
    connect(m_pulseTimer, &QTimer::timeout, this, [this](){ tickDots(); });
}


void ProgressSplash::start(const QString& title)
{
    m_baseTitle = title.isEmpty() ? QStringLiteral("Reconstructing") : title;
    m_pulseStep = 0;

    if (m_title) m_title->setText(m_baseTitle);
    if (m_dots)  m_dots->setText(QString());
    if (m_bar)   m_bar->setValue(0);

    centerOnOwner();
    show(); raise(); activateWindow();
    if (m_pulseTimer && !m_pulseTimer->isActive()) m_pulseTimer->start();

    qDebug() << "[SPLASH] start baseTitle=" << m_baseTitle;
}

void ProgressSplash::updateProgress(int pct, const QString& stage)
{
    if (m_bar) m_bar->setValue(qBound(0, pct, 100));
    qDebug() << "[SPLASH] update:" << pct << "(stage ignored visually =" << stage << ")";
    centerOnOwner(); // keep centered if owner moved
}

void ProgressSplash::finish()
{
    qDebug() << "[SPLASH] finish()";
    if (m_pulseTimer) m_pulseTimer->stop();
    hide();
    deleteLater();
}

bool ProgressSplash::eventFilter(QObject* obj, QEvent* ev)
{
    if (obj == m_owner && (ev->type() == QEvent::Move || ev->type() == QEvent::Resize || ev->type() == QEvent::Show))
        centerOnOwner();
    return QDialog::eventFilter(obj, ev);
}

void ProgressSplash::centerOnOwner()
{
    QRect target;
    if (m_owner) {
        const QPoint gpos = m_owner->mapToGlobal(QPoint(0, 0));
        target = QRect(gpos, m_owner->size());
    } else {
        target = QApplication::primaryScreen()->availableGeometry();
    }

    const QSize sz = size();
    const int x = target.x() + (target.width()  - sz.width())  / 2;
    const int y = target.y() + (target.height() - sz.height()) / 2;
    setGeometry(QRect(QPoint(x, y), sz));

    if (m_card) m_card->setGeometry(rect());
}

void ProgressSplash::tickDots()
{
    // Separate line animation: "", " .", " . .", " . . ."
    QString dots;
    if (m_pulseStep == 1) dots = " .";
    else if (m_pulseStep == 2) dots = " . .";
    else if (m_pulseStep == 3) dots = " . . .";

    if (m_dots) m_dots->setText(dots);
    qDebug() << "[SPLASH][pulse]" << (m_dots ? m_dots->text() : QString());

    m_pulseStep = (m_pulseStep + 1) & 3; // 0..3 wrap
}
