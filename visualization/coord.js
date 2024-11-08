export function coords(dados) {
    // Extrair valores para cada variável
    //console.log(dados)
    // Extrair valores para cada variável
    const variavel1 = dados.map(d => d.Variavel1);
    const variavel2 = dados.map(d => d.Variavel2);
    const variavel3 = dados.map(d => d.Variavel3);

    const trace = {
        type: 'parcoords',
        line: {
            color: variavel1, // Cor das linhas com base em Variavel1
            colorscale: 'Viridis', // Mapa de cores
            showscale: true
        },
        dimensions: [
            {
                label: 'Variável 1',
                values: variavel1
            },
            {
                label: 'Variável 2',
                values: variavel2
            },
            {
                label: 'Variável 3',
                values: variavel3
            }
        ]
    };

    const layout = {
        title: 'Gráfico de Coordenadas Paralelas',
        height: 400
    };

    // Renderizar o gráfico
    Plotly.newPlot('coord', [trace], layout);
}
