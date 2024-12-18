export default function playVideo(metrics) {
  const videoOriginal = metrics.video_url_nao_trackeado;
  const videoTrackeado = metrics.video_url_trackeado;

  // Divide o caminho em partes usando "\\" e obtém os dois últimos segmentos
  const partsOriginal = videoOriginal.split("\\").slice(-2).join("\\");
  const srcvideoOriginal = `outputs\\${partsOriginal}`; //videoOriginal;
  const partsTrackeado = videoTrackeado.split("\\").slice(-2).join("\\");
  const srcvideoTrackeado = `outputs\\${partsTrackeado}`; //videoTrackeado;
  
  console.log(metrics.id);
  console.log(srcvideoOriginal);
  console.log(srcvideoTrackeado);

  // Referências aos elementos de vídeo
  const videoElement = d3.select("#video");
  const trackerElement = d3.select("#video-tracker");

  // Atualizar ou criar a fonte para o vídeo original
  let originalSource = videoElement.select("source");
  if (originalSource.empty()) {
      originalSource = videoElement.append("source").attr("type", "video/mp4");
  }
  originalSource.attr("src", srcvideoOriginal);

  // Atualizar ou criar a fonte para o vídeo trackeado
  let trackerSource = trackerElement.select("source");
  if (trackerSource.empty()) {
      trackerSource = trackerElement.append("source").attr("type", "video/mp4");
  }
  trackerSource.attr("src", srcvideoTrackeado);

  // Recarregar os vídeos
  document.getElementById("video").load();
  document.getElementById("video-tracker").load();
}
