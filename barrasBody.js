export async function barrasBody(id) {
    console.log("Iniciando carregamento dos dados");

    try {
        const data = await d3.csv("outputs/df_grouped.csv");

        // Definir faixas de IMC
        const bins = [0, 18.5, 25, 30, 35, 40, 100];
        const labels = ['Abaixo do peso', 'Peso normal', 'Sobrepeso', 'Obesidade Grau I', 'Obesidade Grau II', 'Obesidade Grau III'];

        // Agrupar e processar os dados por IMC
        let groupedData = {};
        data.forEach(d => {
            let bmi = +d['Body mass index (kg/m²)'];
            let group = labels[bins.findIndex((b, i) => bmi >= b && bmi < bins[i + 1]) - 1];

            if (!groupedData[group]) {
                groupedData[group] = { 'BMI_Group': group, 'Tipo_A': 0, 'Tipo_B': 0, 'Tipo_C': 0, 'Tipo_D': 0, 'Hiperativo': 0, 'Total': 0 };
            }

            ['Tipo_A', 'Tipo_B', 'Tipo_C', 'Tipo_D', 'Hiperativo'].forEach(tipo => {
                groupedData[group][tipo] += +d[tipo];
                groupedData[group]['Total'] += +d[tipo];
            });
        });

        // Calcular proporções
        const finalData = labels.map(label => {
            const group = groupedData[label] || { 'Tipo_A': 0, 'Tipo_B': 0, 'Tipo_C': 0, 'Tipo_D': 0, 'Hiperativo': 0, 'Total': 1 };

            return {
                BMI_Group: label,
                Tipo_A: group['Tipo_A'] / group['Total'],
                Tipo_B: group['Tipo_B'] / group['Total'],
                Tipo_C: group['Tipo_C'] / group['Total'],
                Tipo_D: group['Tipo_D'] / group['Total'],
                Hiperativo: group['Hiperativo'] / group['Total']
            };
        });

        const idBarras = data.find(d => d['ID'] == id);

        // Dados do usuário
        const dadosUsuario = {
            Tipo_A: parseFloat(idBarras['Tipo_A']),
            Tipo_B: parseFloat(idBarras['Tipo_B']),
            Tipo_C: parseFloat(idBarras['Tipo_C']),
            Tipo_D: parseFloat(idBarras['Tipo_D']),
            Hiperativo: parseFloat(idBarras['Hiperativo']),
            bmi: idBarras['Body mass index (kg/m²)']
        };

        console.log(dadosUsuario);
        // Encontrar faixa de IMC do usuário
        const userBMIGroup = labels[bins.findIndex((b, i) => dadosUsuario.bmi >= b && dadosUsuario.bmi < bins[i + 1])];

        // Função para determinar seta
        const getArrow = (userValue, groupValue) => userValue > groupValue ? '↑' : '↓';

        // Configurar as barras para Plotly
        const xValues = labels;
        const traces = ['Tipo_A', 'Tipo_B', 'Tipo_C', 'Tipo_D', 'Hiperativo'].map(tipo => {
            return {
                x: xValues,
                y: finalData.map(d => d[tipo]),
                name: tipo,
                type: 'bar',
                text: finalData.map(d => (d[tipo] * 100).toFixed(1) + '%'),
                textposition: 'inside'
            };
        });

        // Comparar valores do usuário com médias e criar anotações
        const annotations = ['Tipo_A', 'Tipo_B', 'Tipo_C', 'Tipo_D', 'Hiperativo'].map((tipo, i) => {
            const userValue = (dadosUsuario[tipo]) / (dadosUsuario['Tipo_A'] + dadosUsuario['Tipo_B'] + dadosUsuario['Tipo_C'] + dadosUsuario['Tipo_D'] + dadosUsuario['Hiperativo']);
            const valorPadrao = finalData.find(d => d.BMI_Group === userBMIGroup);
            const groupValue = valorPadrao[tipo];
            let groupValue2 = 0;
            if (tipo == 'Tipo_A') {
                groupValue2 = 0;

            } else if (tipo == 'Tipo_B') {
                groupValue2 = valorPadrao['Tipo_A'] + valorPadrao['Tipo_B']/2;
            } else if (tipo == 'Tipo_C') {
                groupValue2 = valorPadrao['Tipo_A'] + valorPadrao['Tipo_B']+ valorPadrao['Tipo_C']/2;
            } else if (tipo == 'Tipo_D') {
                groupValue2 = valorPadrao['Tipo_A'] + valorPadrao['Tipo_B'] + valorPadrao['Tipo_C'] + valorPadrao['Tipo_D']/2;
            } else {
                groupValue2 = valorPadrao['Tipo_A'] + valorPadrao['Tipo_B'] + valorPadrao['Tipo_C'] + valorPadrao['Tipo_D']+ valorPadrao['Hiperativo']/2;

            }

            return {
                x: userBMIGroup,
                y: groupValue2,
                text: getArrow(userValue, groupValue),
                showarrow: false,
                font: { size: 20, weight: '2', color: 'white' }
            };
        });

        // Layout do gráfico
        const layout = {
            barmode: 'stack',
            width: '450',
            height: '300',
            title: 'IMC x Tipo de espermatozóide',
            xaxis: { title: 'Faixa de IMC' },
            yaxis: { title: 'Proporção (%)', tickformat: ',.0%', range: [0, 1] },
            annotations: annotations,
            margin: { t: 50, r: 0, b: 50, l: 50 },
            font: {
                size: 9  // Tamanho da fonte geral do gráfico
            }
        };

        // Renderizar o gráfico
        Plotly.newPlot('barrasBody', traces, layout);

        console.log("Gráfico gerado com sucesso!");
    } catch (error) {
        console.error('Erro ao carregar o CSV:', error);
    }
}
