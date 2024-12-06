import { glyph_by_frame } from './glyph_tracker.js';
import playVideo from './video.js';
import { coords } from './coord.js';



// Carrega os dados do JSON
let variaveis = await d3.json("data.json");

let dados = await d3.csv("data_horm_concat_corr.csv");

// Função para atualizar a visualização com base no ID do indivíduo
function atualizarVisualizacao(id) {
    // Encontra o indivíduo específico pelo ID
    let individuo = variaveis.individuos.find(ind => ind.id === id);

    if (individuo) {
        // Atualiza o gráfico com os dados do indivíduo selecionado
        glyph_by_frame(individuo.espermatozoides);
        playVideo();
        coords(dados, id);
    }

}

// Cria o dropdown com os IDs dos indivíduos
d3.select("#individuos")
    .selectAll("option")
    .data(variaveis.individuos)
    .enter()
    .append("option")
    .attr("value", d => d.id)
    .text(d => `Indivíduo ${d.id}`);

// Adiciona evento para quando o usuário selecionar um indivíduo no dropdown
d3.select("#individuos").on("change", function () {
    const idSelecionado = +this.value;
    atualizarVisualizacao(idSelecionado);
});

// Chama a função com um ID padrão (ex. 1) para mostrar a visualização inicial
atualizarVisualizacao(1);
// main.js
