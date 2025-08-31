const { app, BrowserWindow, Menu, dialog } = require('electron');
const { autoUpdater } = require('electron-updater');
const path = require('path');

let mainWindow;

// Auto-updater configuration
autoUpdater.checkForUpdatesAndNotify();

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        frame: false,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
        icon: path.join(__dirname, 'icon.ico'),
        title: 'AiBeatzbyJyntzu - Professional AI Beat Maker',
        show: false
    });

    mainWindow.loadFile('complete_desktop_app.html');
    
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        
        // Check for updates on startup
        if (!process.env.NODE_ENV || process.env.NODE_ENV === 'production') {
            autoUpdater.checkForUpdatesAndNotify();
        }
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Remove default menu
    Menu.setApplicationMenu(null);
}

// Auto-updater events
autoUpdater.on('checking-for-update', () => {
    console.log('Checking for update...');
});

autoUpdater.on('update-available', (info) => {
    console.log('Update available.');
    dialog.showMessageBox(mainWindow, {
        type: 'info',
        title: 'Update Available',
        message: `AiBeatzbyJyntzu v${info.version} is available!`,
        detail: 'The update will be downloaded in the background. You will be notified when it is ready to install.',
        buttons: ['OK']
    });
});

autoUpdater.on('update-not-available', (info) => {
    console.log('Update not available.');
});

autoUpdater.on('error', (err) => {
    console.log('Error in auto-updater. ' + err);
});

autoUpdater.on('download-progress', (progressObj) => {
    let log_message = "Download speed: " + progressObj.bytesPerSecond;
    log_message = log_message + ' - Downloaded ' + progressObj.percent + '%';
    log_message = log_message + ' (' + progressObj.transferred + "/" + progressObj.total + ')';
    console.log(log_message);
});

autoUpdater.on('update-downloaded', (info) => {
    console.log('Update downloaded');
    dialog.showMessageBox(mainWindow, {
        type: 'info',
        title: 'Update Ready',
        message: `AiBeatzbyJyntzu v${info.version} has been downloaded and is ready to install.`,
        detail: 'The application will restart to apply the update.',
        buttons: ['Restart Now', 'Later']
    }).then((result) => {
        if (result.response === 0) {
            autoUpdater.quitAndInstall();
        }
    });
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

// Check for updates every hour
setInterval(() => {
    autoUpdater.checkForUpdatesAndNotify();
}, 60 * 60 * 1000);