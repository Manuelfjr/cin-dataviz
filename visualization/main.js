import { glyph } from './glyph.js';
import playVideo from './video.js';
import { coords } from './coord.js';



// Carrega os dados do JSON
let variaveis = await d3.json("data.json");
const dados = [
    { Variavel1: 1, Variavel2: 2, Variavel3: 3 },
    { Variavel1: 2, Variavel2: 1, Variavel3: 2 },
    { Variavel1: 3, Variavel2: 5, Variavel3: 3 },
    { Variavel1: 4, Variavel2: 2, Variavel3: 1 },
    { Variavel1: 5, Variavel2: 3, Variavel3: 4 }
];

// Função para atualizar a visualização com base no ID do indivíduo
function atualizarVisualizacao(id) {
    // Encontra o indivíduo específico pelo ID
    let individuo = variaveis.individuos.find(ind => ind.id === id);

    if (individuo) {
        // Atualiza o gráfico com os dados do indivíduo selecionado
        glyph(individuo.espermatozoides);
        playVideo();
        coords(dados);
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
