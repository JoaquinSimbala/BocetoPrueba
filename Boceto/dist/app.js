"use strict";
const ui = {
    video: document.getElementById("video"),
    videoHidden: document.getElementById("videoHidden"),
    overlay: document.getElementById("overlay"),
    status: document.getElementById("status"),
    infoBtn: document.getElementById("infoBtn"),
    closeModalBtn: document.getElementById("closeModalBtn"),
    infoModal: document.getElementById("infoModal"),
    modelInfo: document.getElementById("modelInfo"),
    prediction: document.getElementById("prediction"),
    recyclableResult: document.getElementById("recyclableResult"),
    simulateBtn: document.getElementById("simulateBtn"),
    flipBtn: document.getElementById("flipBtn"),
    alert: document.getElementById("alert")
};
let stream = null;
let isAnalyzing = false;
let model = null;
const recyclableClasses = ["bottle", "cup", "can", "book", "tv", "laptop", "cell phone"];
/**
 * Actualiza el estado visual del overlay y el texto de estado de cámara/análisis.
 */
function setVisualState(state, text) {
    ui.overlay.classList.remove("processing", "success", "error");
    ui.overlay.classList.add(state);
    ui.status.textContent = text;
}
/**
 * Muestra un mensaje de alerta en la sección inferior.
 */
function setAlert(message) {
    ui.alert.textContent = message;
}
/**
 * Refleja en UI el estado de carga/disponibilidad del modelo de IA.
 */
function setModelInfo(text) {
    ui.modelInfo.textContent = text;
}
/**
 * Abre el modal de información del análisis.
 */
function openModal() {
    ui.infoModal.classList.add("is-open");
    ui.infoModal.setAttribute("aria-hidden", "false");
}
/**
 * Cierra el modal de información del análisis.
 */
function closeModal() {
    ui.infoModal.classList.remove("is-open");
    ui.infoModal.setAttribute("aria-hidden", "true");
}
/**
 * Construye los parámetros de acceso a cámara con resolución moderada
 * y preferencia por cámara trasera en móviles.
 */
function getConstraints() {
    return {
        audio: false,
        video: {
            width: { ideal: 960, max: 1280 },
            height: { ideal: 540, max: 720 },
            facingMode: { ideal: "environment" }
        }
    };
}
/**
 * Solicita acceso a la cámara. Si los constraints fallan, aplica fallback.
 */
async function requestCamera() {
    try {
        return await navigator.mediaDevices.getUserMedia(getConstraints());
    }
    catch (err) {
        if (err.name === "OverconstrainedError") {
            return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
        throw err;
    }
}
/**
 * Traduce errores técnicos de cámara a mensajes legibles para usuario.
 */
function explainError(err) {
    const errorName = err?.name;
    if (errorName === "NotAllowedError") {
        return "Permiso denegado. Activa la camara en tu navegador.";
    }
    if (errorName === "NotFoundError") {
        return "No se encontro camara en este dispositivo.";
    }
    if (errorName === "NotReadableError") {
        return "La camara esta siendo usada por otra aplicacion.";
    }
    return "No fue posible iniciar la camara. Recarga e intenta otra vez.";
}
/**
 * Cambia el estado del botón principal (habilitado/procesando) sin
 * reemplazar su icono.
 */
function setButtonState(disabled, isProcessing = false) {
    ui.simulateBtn.disabled = disabled;
    ui.simulateBtn.classList.toggle("is-processing", isProcessing);
    if (isProcessing) {
        ui.simulateBtn.setAttribute("aria-label", "Procesando");
        ui.simulateBtn.setAttribute("title", "Procesando");
        return;
    }
    ui.simulateBtn.setAttribute("aria-label", "Analizar con IA");
    ui.simulateBtn.setAttribute("title", "Analizar");
}
/**
 * Evalúa si la clase detectada se considera reciclable según una lista base.
 */
function isRecyclableClass(className) {
    const normalized = className.toLowerCase();
    return recyclableClasses.includes(normalized);
}
/**
 * Inicializa TensorFlow.js, selecciona backend y carga el modelo coco-ssd.
 */
async function loadTfModel() {
    setModelInfo("Modelo IA: preparando TensorFlow.js...");
    try {
        await tf.setBackend("webgl");
    }
    catch {
        await tf.setBackend("cpu");
    }
    await tf.ready();
    setModelInfo("Modelo IA: cargando coco-ssd...");
    model = await cocoSsd.load();
    setModelInfo("Modelo IA: listo");
}
/**
 * Ejecuta una inferencia sobre el frame actual del video y actualiza la UI
 * con predicción y estado de reciclabilidad.
 */
async function simulateOnce() {
    if (!stream || !model || isAnalyzing) {
        return;
    }
    isAnalyzing = true;
    setButtonState(true, true);
    setVisualState("processing", "Procesando...");
    try {
        const predictions = await model.detect(ui.videoHidden);
        const best = predictions[0];
        if (!best) {
            ui.prediction.textContent = "Prediccion: sin objeto detectado";
            ui.recyclableResult.textContent = "No";
            setVisualState("error", "No se detecto objeto");
            setAlert("Coloca el residuo dentro del recuadro y vuelve a analizar.");
            return;
        }
        const confidence = Math.round(best.score * 100);
        const recyclable = isRecyclableClass(best.class);
        ui.prediction.textContent = `Prediccion: ${best.class} - ${confidence}% certeza`;
        ui.recyclableResult.textContent = recyclable ? "Si" : "No";
        if (recyclable) {
            setVisualState("success", "Reciclable detectado");
            setAlert("");
        }
        else {
            setVisualState("error", "No reciclable detectado");
            setAlert("Sugerencia: verifica el tipo de residuo antes de desecharlo.");
        }
    }
    catch {
        setVisualState("error", "Fallo de inferencia IA");
        setAlert("No se pudo analizar el frame con TensorFlow.js.");
    }
    finally {
        setButtonState(false);
        isAnalyzing = false;
    }
}
/**
 * Orquesta el flujo inicial: solicita permisos de cámara, reproduce video,
 * carga modelo y habilita el botón de análisis.
 */
async function init() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setVisualState("error", "Navegador sin soporte");
        setAlert("Este navegador no soporta getUserMedia.");
        return;
    }
    try {
        setVisualState("processing", "Solicitando permiso...");
        setAlert("");
        stream = await requestCamera();
        ui.video.srcObject = stream;
        ui.videoHidden.srcObject = stream;
        await ui.video.play();
        await loadTfModel();
        setVisualState("success", "Camara activa");
        setButtonState(false);
    }
    catch (err) {
        setVisualState("error", "Error de camara");
        setAlert(explainError(err));
        setModelInfo("Modelo IA: no disponible");
        setButtonState(true);
        console.error(err);
    }
}
ui.simulateBtn.addEventListener("click", () => {
    void simulateOnce();
});
ui.infoBtn.addEventListener("click", () => {
    openModal();
});
ui.closeModalBtn.addEventListener("click", () => {
    closeModal();
});
ui.infoModal.addEventListener("click", (event) => {
    const target = event.target;
    if (target.hasAttribute("data-close-modal")) {
        closeModal();
    }
});
ui.flipBtn.addEventListener("click", () => {
});
void init();
