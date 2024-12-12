import { glyph_by_frame } from './glyph_tracker.js';
import { glyph } from './glyph.js';
import playVideo from './video.js';
import { coords } from './coord.js';

function downloadFile(url, filename) {
    // Create an anchor element
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;

    // Append the anchor to the body
    document.body.appendChild(a);

    // Programmatically click the anchor
    a.click();

    // Remove the anchor from the body
    document.body.removeChild(a);
}

// Carrega os dados do JSON
// URL de download

const variaveis = await d3.json("outputs/data_window.json");
const metrics_gerais = await d3.json("metrics_general.json");
const dados = await d3.csv("outputs/data_horm_concat_corr.csv");

// Função para atualizar a visualização com base no ID do indivíduo
function atualizarVisualizacao(id) {
    // Encontra o indivíduo específico pelo ID
    const individuo = variaveis.individuos.find(ind => ind.id === id);
    const metrics = metrics_gerais.metrics.find(ind => ind.id === id);

    if (individuo && metrics) {
        // Atualiza o gráfico com os dados do indivíduo selecionado
        glyph_by_frame(individuo, metrics);
        playVideo(metrics);
        coords(dados, id);
    }
}

// Função para atualizar o dropdown de espermatozoides com base no indivíduo selecionado
function atualizarDropdownEspermatozoides(individuo) {
    const dropdownEspermatozoides = d3.select("#espermatozoides");

    // Limpa opções existentes
    dropdownEspermatozoides.selectAll("option").remove();

    // Adiciona novas opções
    dropdownEspermatozoides
        .selectAll("option")
        .data(individuo.espermatozoides)
        .enter()
        .append("option")
        .attr("value", d => d.id)
        .text(d => `Esperm. ${d.id}`);
}

// Cria o dropdown com os IDs dos indivíduos
d3.select("#individuos")
    .selectAll("option")
    .data(variaveis.individuos)
    .enter()
    .append("option")
    .attr("value", d => d.id)
    .text(d => `Indivíduo ${d.id}`);

// Evento para quando o usuário selecionar um indivíduo no dropdown
d3.select("#individuos").on("change", function () {
    const idSelecionado = +this.value; // Obtém o ID do indivíduo selecionado
    const individuo = variaveis.individuos.find(ind => ind.id === idSelecionado);

    if (individuo) {
        atualizarDropdownEspermatozoides(individuo); // Atualiza o dropdown de espermatozoides
        atualizarVisualizacao(idSelecionado); // Atualiza a visualização
    }
});

// Evento para quando o usuário selecionar um espermatozoide no dropdown
d3.select("#espermatozoides").on("change", function () {
    const idEsp = +this.value; // Obtém o ID do espermatozoide selecionado
    const idInd = +d3.select("#individuos").node().value; // Obtém o ID do indivíduo atualmente selecionado

    const metrics = metrics_gerais.metrics.find(ind => ind.id === idInd);
    const espMetrics = metrics?.trackers?.find(esp => esp.tracker_id === idEsp);

    if (espMetrics) {
        glyph(espMetrics); // Atualiza o glifo do espermatozoide
    }
});

// Inicializa com um ID padrão
const idPadrao = 1;
const idEspPadrao = 0;
const individuoPadrao = variaveis.individuos.find(ind => ind.id === idPadrao);

const metrics = metrics_gerais.metrics.find(ind => ind.id === idPadrao);
const esp_metrics = metrics.trackers.find(esp => esp.tracker_id === idEspPadrao);

if (individuoPadrao) {
    atualizarDropdownEspermatozoides(individuoPadrao);
    atualizarVisualizacao(idPadrao);
    glyph(esp_metrics);
}
