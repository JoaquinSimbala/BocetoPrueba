type ScanState = "processing" | "success" | "error";

type CocoPrediction = {
    class: string;
    score: number;
    bbox: [number, number, number, number];
};

type CocoSsdModel = {
    detect(input: HTMLVideoElement): Promise<CocoPrediction[]>;
};

declare const tf: {
    setBackend(backendName: string): Promise<void>;
    ready(): Promise<void>;
};

declare const cocoSsd: {
    load(): Promise<CocoSsdModel>;
};

type UiRefs = {
    video: HTMLVideoElement;
    videoHidden: HTMLVideoElement;
    overlay: HTMLDivElement;
    status: HTMLDivElement;
    infoBtn: HTMLButtonElement;
    closeModalBtn: HTMLButtonElement;
    infoModal: HTMLDivElement;
    modelInfo: HTMLParagraphElement;
    prediction: HTMLParagraphElement;
    recyclableResult: HTMLParagraphElement;
    simulateBtn: HTMLButtonElement;
    flipBtn: HTMLButtonElement;
    alert: HTMLParagraphElement;
};

const ui: UiRefs = {
    video: document.getElementById("video") as HTMLVideoElement,
    videoHidden: document.getElementById("videoHidden") as HTMLVideoElement,
    overlay: document.getElementById("overlay") as HTMLDivElement,
    status: document.getElementById("status") as HTMLDivElement,
    infoBtn: document.getElementById("infoBtn") as HTMLButtonElement,
    closeModalBtn: document.getElementById("closeModalBtn") as HTMLButtonElement,
    infoModal: document.getElementById("infoModal") as HTMLDivElement,
    modelInfo: document.getElementById("modelInfo") as HTMLParagraphElement,
    prediction: document.getElementById("prediction") as HTMLParagraphElement,
    recyclableResult: document.getElementById("recyclableResult") as HTMLParagraphElement,
    simulateBtn: document.getElementById("simulateBtn") as HTMLButtonElement,
    flipBtn: document.getElementById("flipBtn") as HTMLButtonElement,
    alert: document.getElementById("alert") as HTMLParagraphElement
};

let stream: MediaStream | null = null;
let isAnalyzing = false;
let model: CocoSsdModel | null = null;

const recyclableClasses = ["bottle", "cup", "can", "book", "tv", "laptop", "cell phone"];

/**
 * Actualiza el estado visual del overlay y el texto de estado de cámara/análisis.
 */
function setVisualState(state: ScanState, text: string): void {
    ui.overlay.classList.remove("processing", "success", "error");
    ui.overlay.classList.add(state);
    ui.status.textContent = text;
}

/**
 * Muestra un mensaje de alerta en la sección inferior.
 */
function setAlert(message: string): void {
    ui.alert.textContent = message;
}

/**
 * Refleja en UI el estado de carga/disponibilidad del modelo de IA.
 */
function setModelInfo(text: string): void {
    ui.modelInfo.textContent = text;
}

/**
 * Abre el modal de información del análisis.
 */
function openModal(): void {
    ui.infoModal.classList.add("is-open");
    ui.infoModal.setAttribute("aria-hidden", "false");
}

/**
 * Cierra el modal de información del análisis.
 */
function closeModal(): void {
    ui.infoModal.classList.remove("is-open");
    ui.infoModal.setAttribute("aria-hidden", "true");
}

/**
 * Construye los parámetros de acceso a cámara con resolución moderada
 * y preferencia por cámara trasera en móviles.
 */
function getConstraints(): MediaStreamConstraints {
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
async function requestCamera(): Promise<MediaStream> {
    try {
        return await navigator.mediaDevices.getUserMedia(getConstraints());
    } catch (err) {
        if ((err as DOMException).name === "OverconstrainedError") {
            return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
        throw err;
    }
}

/**
 * Traduce errores técnicos de cámara a mensajes legibles para usuario.
 */
function explainError(err: unknown): string {
    const errorName = (err as DOMException)?.name;

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
function setButtonState(disabled: boolean, isProcessing = false): void {
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
function isRecyclableClass(className: string): boolean {
    const normalized = className.toLowerCase();
    return recyclableClasses.includes(normalized);
}

/**
 * Inicializa TensorFlow.js, selecciona backend y carga el modelo coco-ssd.
 */
async function loadTfModel(): Promise<void> {
    setModelInfo("Modelo IA: preparando TensorFlow.js...");

    try {
        await tf.setBackend("webgl");
    } catch {
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
async function simulateOnce(): Promise<void> {
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
        } else {
            setVisualState("error", "No reciclable detectado");
            setAlert("Sugerencia: verifica el tipo de residuo antes de desecharlo.");
        }
    } catch {
        setVisualState("error", "Fallo de inferencia IA");
        setAlert("No se pudo analizar el frame con TensorFlow.js.");
    } finally {
        setButtonState(false);
        isAnalyzing = false;
    }
}

/**
 * Orquesta el flujo inicial: solicita permisos de cámara, reproduce video,
 * carga modelo y habilita el botón de análisis.
 */
async function init(): Promise<void> {
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
    } catch (err) {
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
    const target = event.target as HTMLElement;
    if (target.hasAttribute("data-close-modal")) {
        closeModal();
    }
});

ui.flipBtn.addEventListener("click", () => {
});

window.addEventListener("beforeunload", () => {
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
    }
});

void init();
