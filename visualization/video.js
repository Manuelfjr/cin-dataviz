export default function playVideo() {
  const videoUrl = "11.2.mp4"; // Substitua pelo caminho do seu vídeo

  // Seleciona o elemento de vídeo e define o source com D3.js
  d3.selectAll("#source").remove();
  d3.select(".container")
    .select(".video")
    .append("source")
    .attr("width", 490) // Define a largura do vídeo
    .attr("height", 380)
    .attr("src", videoUrl)
    .attr("type", "video/mp4")
}