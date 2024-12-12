export function coords(dados, individuo) {
    // Extrair valores para cada variável
    const id = dados.map(d => d['ID']);
    const spermC17_0 = dados.map(d => parseFloat(d['Sperm C17:0']));
    const totalSpermCount = dados.map(d => parseFloat(d['Total sperm count']));
    const spermC24_0 = dados.map(d => parseFloat(d['Sperm C24:0']));
    const spermC18_1_trans = dados.map(d => parseFloat(d['Sperm C18:1 trans']));
    const spermC16_1_n7 = dados.map(d => parseFloat(d['Sperm C16:1 n-7']));
    const spermC18_3_n3 = dados.map(d => parseFloat(d['Sperm C18:3 n-3']));
    const spermC18_1_n9 = dados.map(d => parseFloat(d['Sperm C18:1 n-9']));
    const spermC16_0 = dados.map(d => parseFloat(d['Sperm C16:0']));
    const spermC20_1_n9 = dados.map(d => parseFloat(d['Sperm C20:1 n-9']));
    const spermC22_6_n3 = dados.map(d => parseFloat(d['Sperm C22:6 n-3']));
    const age = dados.map(d => parseFloat(d['Age']));
    const bmi = dados.map(d => parseFloat(d['BMI']));
    const spermVitality = dados.map(d => parseFloat(d['Sperm vitality']));

    // Criar uma lista de cores, onde todas as linhas têm cor 0, exceto a linha do "individuo"
    const colors = id.map(val => (val == individuo ? 1 : 0)); // 1 para a linha selecionada, 0 para as outras

    // Criar uma escala de cores para usar cinza e vermelho
    // const colorScale = [[0, 'rgb(150,150,150)'], [1, 'rgb(255,0,0)']];
    const colorScale = [[0, 'rgb(96, 96, 221)'], [1, 'rgb(255,0,0)']];

    // Criar o trace (dados) do gráfico
    const trace = {
        type: 'parcoords',
        line: {
            color: colors, // Usando a lista de cores (1 para vermelho, 0 para cinza)
            colorscale: colorScale, // Define a escala de cores personalizada
            showscale: false   // Mostrar escala de cores

        },
        labelangle: -25,
        dimensions: [
            { label: 'ID', values: id },
            { label: 'Sperm C17:0', values: spermC17_0, tickangle: 45 },
            { label: 'Total sperm count', values: totalSpermCount },
            { label: 'Sperm C24:0', values: spermC24_0 },
            { label: 'Sperm C18:1 trans', values: spermC18_1_trans },
            { label: 'Sperm C16:1 n-7', values: spermC16_1_n7 },
            { label: 'Sperm C18:3 n-3', values: spermC18_3_n3 },
            { label: 'Sperm C18:1 n-9', values: spermC18_1_n9 },
            { label: 'Sperm C16:0', values: spermC16_0 },
            { label: 'Sperm C20:1 n-9', values: spermC20_1_n9 },
            { label: 'Sperm C22:6 n-3', values: spermC22_6_n3 },
            { label: 'Age', values: age },
            { label: 'BMI', values: bmi },
            { label: 'Sperm vitality', values: spermVitality }
        ]
    };

    // Layout do gráfico
    let div_coord = document.querySelector('#coord');
    let largura = div_coord.clientWidth;
    const layout = {
        height: 300, // Ajuste da altura para o gráfico
        margin: {
            l: 40, // Margem esquerda
            r: 40, // Margem direita
            t: 80, // Margem superior
            b: 10  // Margem inferior
        },
        width: largura - 5
    };

    // Renderizar o gráfico
    Plotly.newPlot('coord', [trace], layout);
}
