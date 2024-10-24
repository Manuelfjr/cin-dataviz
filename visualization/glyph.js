let variaveis = await d3.json("data.json");
let posicoes = [{'x':'300', 'y': '300'},
                {'x':'900', 'y': '300'}
]

const svg = d3.select("svg");
const centerX = variaveis.centerX;
const centerY = variaveis.centerY;

// Raio dos semicírculos
const radii = {
    outer: variaveis.radii.outer,
    middle: variaveis.radii.middle,
    inner: variaveis.radii.inner,
};

// Fenda de 30 graus
let startAngle = -5 * Math.PI / 6; // -30 graus
let endAngle = 5 * Math.PI / 6;    // 30 graus

// Função para criar semicírculos
function drawSemiCircle(innerRadius, outerRadius, stroke, fill = "none", strokeWidth = 3, startAngle, endAngle) {
    svg.append("path")
        .attr("d", d3.arc()
            .innerRadius(innerRadius)
            .outerRadius(outerRadius)
            .startAngle(startAngle)
            .endAngle(endAngle))
        .attr("fill", fill)
        .attr("stroke", stroke)
        .attr("stroke-width", strokeWidth)
        .attr("transform", `translate(${centerX}, ${centerY})`);
}

// Desenhar semicírculos
drawSemiCircle(radii.outer, radii.outer, "gray", "none", 3, startAngle, endAngle);
drawSemiCircle(radii.middle, radii.middle, "black", "none", 5, startAngle, endAngle);
drawSemiCircle(radii.inner, radii.inner, "gray", "none", 3, startAngle, endAngle);
drawSemiCircle(radii.inner, radii.inner - 20, "none", 'rgba(150, 0, 150, 0.8)', 0, startAngle, endAngle); // Semicírculo interno roxo
drawSemiCircle(radii.outer, radii.outer + 20, "none", 'rgba(150, 150, 150, 0.4)', 0, startAngle, -1); // Semicírculo externo cinza
drawSemiCircle(radii.inner - 20, 0, "none", 'rgba(150, 150, 150, 0.4)', 0, -Math.PI, Math.PI); // Semicírculo interno roxo
drawSemiCircle(radii.inner - 20, 0, "gray", 'white', 2, Math.PI / 6, -Math.PI / 6); // Semicírculo interno roxo

const lineLength = variaveis.lineLength;

// Função para desenhar triângulos
function drawTriangle(points, fill, stroke, strokeWidth) {
    svg.append("polygon")
        .attr("points", points.map(p => `${p.x},${p.y}`).join(" "))
        .attr("fill", fill)
        .attr("stroke", stroke)
        .attr("stroke-width", strokeWidth);
}

// Triângulos da base
const baseTriangle = [
    { x: centerX, y: centerY + 80 },
    { x: centerX - (lineLength / 6), y: centerY + 180 },
    { x: centerX + (lineLength / 6), y: centerY + 180 }
];
drawTriangle(baseTriangle, "rgba(100, 100, 100, 0.5)", "rgba(100, 100, 100)", 2);

const topTriangle = [
    { x: centerX, y: centerY - 240 },
    { x: centerX - (lineLength / 7), y: centerY - 205 },
    { x: centerX + (lineLength / 7), y: centerY - 205 }
];
drawTriangle(topTriangle, "black", "black", 2);

// Adicionar linhas em cruz
const crossLength = 100; // Comprimento das linhas
svg.append("line") // Linha vertical
    .attr("x1", centerX)
    .attr("y1", centerY)
    .attr("x2", centerX)
    .attr("y2", centerY + 200)
    .attr("stroke", "white")
    .attr("stroke-width", 2);

svg.append("line") // Linha horizontal
    .attr("x1", centerX - 80)
    .attr("y1", centerY)
    .attr("x2", centerX + 80)
    .attr("y2", centerY)
    .attr("stroke", "white")
    .attr("stroke-width", 2);

// Reta com círculos laranjas
svg.append("line")
    .attr("x1", centerX)
    .attr("y1", centerY + 80)
    .attr("x2", centerX)
    .attr("y2", centerY + 250)
    .attr("stroke", "black")
    .attr("stroke-width", 6)
    .attr("transform", `rotate(35, ${centerX}, ${centerY + 80})`);

// Círculos laranjas ao longo da linha
const circlesCount = variaveis.circlesCount;
const circleRadius = variaveis.circleRadius;

for (let i = 1; i <= circlesCount; i++) {
    svg.append("circle")
        .attr("cx", centerX)
        .attr("cy", centerY + 80 * i)
        .attr("r", circleRadius)
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .attr("fill", "orange")
        .attr("transform", `rotate(35, ${centerX}, ${centerY + 80})`);
}

// Elipse no centro
svg.append("ellipse")
    .attr("cx", centerX)
    .attr("cy", centerY)
    .attr("rx", 30)
    .attr("ry", 50)
    .attr("fill", "lightgreen")
    .attr("stroke", "black")
    .attr("stroke-width", 1)
    .attr("transform", `rotate(-25, ${centerX}, ${centerY})`);

// Reta adicional rotacionada e transladada
const length = 30; // Comprimento da reta
const radius = radii.outer; // Raio do círculo externo

function drawLine(angle, color, stroke) {
    let x1 = centerX + radius * Math.cos(angle * Math.PI / 180);
    let y1 = centerY + radius * Math.sin(angle * Math.PI / 180);
    let x2 = centerX + (radius + length) * Math.cos(angle * Math.PI / 180);
    let y2 = centerY + (radius + length) * Math.sin(angle * Math.PI / 180);

    svg.append("line")
        .attr("x1", x1)
        .attr("y1", y1)
        .attr("x2", x2)
        .attr("y2", y2)
        .attr("stroke", color)
        .attr("stroke-width", stroke);
}

drawLine(0, "black", 7);
drawLine(180, "black", 7);
drawLine(150, 'rgba(170, 170, 170)', 6);
drawLine(120, 'rgba(170, 170, 170)', 6);
drawLine(210, 'rgba(170, 170, 170)', 6);