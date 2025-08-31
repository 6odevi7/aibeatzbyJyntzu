// Window control functions
function minimizeWindow() {
    if (window.electronAPI) {
        window.electronAPI.minimize();
    } else {
        console.log('Minimize not available in browser mode');
    }
}

function closeWindow() {
    if (confirm('Close AiBeatzbyJyntzu?')) {
        if (window.electronAPI) {
            window.electronAPI.close();
        } else {
            window.close();
        }
    }
}