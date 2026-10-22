// use the normal swift button callbacks for press and release
(() => {
    let activeButton = null;

    function jogButton(target) {
        const button = target.closest?.("button");
        return button && /^[+-](?:R?[XYZ]|q)$/i.test(button.textContent.trim()) ? button : null;
    }

    function release() {
        if (!activeButton) return;
        activeButton = null;
        const button = [...document.querySelectorAll("button")]
            .find(item => item.textContent.trim() === "Release jog");
        button?.click();
    }

    document.addEventListener("pointerdown", event => {
        const button = jogButton(event.target);
        if (!button || event.button !== 0) return;
        event.preventDefault();
        button.setPointerCapture(event.pointerId);
        activeButton = button;
        button.click();
    }, true);

    // ignore the extra browser click generated after the pointer is released
    document.addEventListener("click", event => {
        if (event.isTrusted && jogButton(event.target)) {
            event.preventDefault();
            event.stopImmediatePropagation();
        }
    }, true);

    document.addEventListener("pointerup", release, true);
    document.addEventListener("pointercancel", release, true);
    document.addEventListener("lostpointercapture", release, true);
    window.addEventListener("blur", release);
    document.addEventListener("focusout", event => {
        if (event.target === activeButton) release();
    });
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) release();
    });

    // keyboard users can hold space or enter on a focused jog button
    document.addEventListener("keydown", event => {
        const button = jogButton(event.target);
        if (!button || ![" ", "Enter"].includes(event.key)) return;
        event.preventDefault();
        if (!event.repeat) {
            activeButton = button;
            button.click();
        }
    }, true);
    document.addEventListener("keyup", event => {
        if (activeButton && [" ", "Enter"].includes(event.key)) {
            event.preventDefault();
            release();
        }
    }, true);
})();
