export default function playVideo() {
  const videoUrl = "11.2.mp4";

  d3.select("#video").selectAll("source").remove();

  // Adiciona um novo elemento source ao vídeo
  d3.select("#video")
    .append("source")
    .attr("src", videoUrl)
    .attr("type", "video/mp4")
}