export const sqrt = Math.sqrt;
export function square(x) {
    return x * x;
}
export function diag(x, y) {
    return sqrt(square(x) + square(y));
}
export function glyph(espermatozoides) {
    d3.selectAll("g").remove(); // remove o que tem antes de desenhar. Para a atualização
    let container = document.querySelector('.container');
    let larguraTela = container.clientWidth;
    let larguraElemento = 280;
    let qtdPorLinha = Math.floor(larguraTela / larguraElemento);
    let altura = larguraElemento * (Math.floor(espermatozoides.length / qtdPorLinha) + (espermatozoides.length % qtdPorLinha !== 0 ? 1 : 0));
    function local(d, i) {
        let x = (i % qtdPorLinha) * larguraElemento;
        let y = Math.floor(i / qtdPorLinha) * larguraElemento;
        return 'translate(' + x + ', ' + y + ')';
    }

    d3.select("svg")
        .attr('width', larguraTela)
        .attr('height', altura)

    d3.select("svg").selectAll("g")
        .data(espermatozoides).enter()
        .append("g")
        .attr('transform', (d, i) => local(d, i));

    const svg = d3.selectAll("g");
    const centerX = 150;
    const centerY = 150;

    const radii = {
        outer: 100,
        middle: 75,
        inner: 50,
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
    let colorScale = d3.scaleSequential()
        .domain([0, 1]) // Ajuste o domínio conforme necessário
        .interpolator(d3.interpolateRainbow);

    // Desenhar semicírculos
    let color = d => colorScale(d.VSL);
    drawSemiCircle(radii.outer, radii.outer, "gray", "none", 1, startAngle, endAngle);
    drawSemiCircle(radii.middle, radii.middle, "black", "none", 3, startAngle, endAngle);
    drawSemiCircle(radii.inner, radii.inner, "gray", "rgba(200, 200, 200)", 1, startAngle, endAngle);
    drawSemiCircle(radii.inner - 10, radii.inner, "none", color, 0, startAngle, endAngle); // Semicírculo interno roxo
    drawSemiCircle(radii.outer, radii.outer + 10, "none", "rgba(200, 200, 200)", 0, startAngle, -1); // Semicírculo externo cinza
    drawSemiCircle(radii.inner - 10, 0, "none", "rgba(200, 200, 200)", 0, -Math.PI, Math.PI); // Semicírculo interno roxo
    drawSemiCircle(radii.inner - 10, 0, "gray", "white", 0.5, Math.PI / 6, -Math.PI / 6); //semicirculo branco
    const lineLength = 75; // 150 / 2

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
        { x: centerX, y: centerY + 40 },
        { x: centerX - (lineLength / 6), y: centerY + 90 },
        { x: centerX + (lineLength / 6), y: centerY + 90 }
    ];
    drawTriangle(baseTriangle, "rgba(100, 100, 100, 0.5)", "rgba(100, 100, 100)", 1);

    const topTriangle = [
        { x: centerX, y: centerY - 120 },
        { x: centerX - (lineLength / 7), y: centerY - 102.5 },
        { x: centerX + (lineLength / 7), y: centerY - 102.5 }
    ];
    drawTriangle(topTriangle, "black", "black", 0);

    // Adicionar linhas em cruz
    const crossLength = 50;
    svg.append("line") // Linha vertical
        .attr("x1", centerX)
        .attr("y1", centerY)
        .attr("x2", centerX)
        .attr("y2", centerY + 100)
        .attr("stroke", "white")
        .attr("stroke-width", 1);

    svg.append("line") // Linha horizontal
        .attr("x1", centerX - 40)
        .attr("y1", centerY)
        .attr("x2", centerX + 40)
        .attr("y2", centerY)
        .attr("stroke", "white")
        .attr("stroke-width", 1);

    // Reta com círculos laranjas
    svg.append("line")
        .attr("x1", centerX)
        .attr("y1", centerY + 40)
        .attr("x2", centerX)
        .attr("y2", centerY + 125)
        .attr("stroke", "black")
        .attr("stroke-width", 4)
        .attr("transform", `rotate(35, ${centerX}, ${centerY + 40})`); // 80 / 2

    // Círculos laranjas ao longo da linha
    const circlesCount = 3;
    const circleRadius = 6;

    for (let i = 1; i <= circlesCount; i++) {
        svg.append("circle")
            .attr("cx", centerX)
            .attr("cy", centerY + 40 * i)
            .attr("r", circleRadius)
            .attr("stroke", "black")
            .attr("stroke-width", 2)
            .attr("fill", "orange")
            .attr("transform", `rotate(35, ${centerX}, ${centerY + 40})`);
    }

    // Elipse no centro
    svg.append("ellipse")
        .attr("cx", centerX)
        .attr("cy", centerY)
        .attr("rx", 15)
        .attr("ry", 25)
        .attr("fill", "lightgreen")
        .attr("stroke", "black")
        .attr("stroke-width", 1)
        .attr("transform", `rotate(-25, ${centerX}, ${centerY})`);

    // Reta adicional rotacionada e transladada
    const length = 15; // 30 / 2
    const radius = radii.outer; // Raio do círculo externo

    function drawLine(angle, color) {
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
            .attr("stroke-width", 4);
    }


    drawLine(0, "black");
    drawLine(180, "black");
    drawLine(150, "rgba(170, 170, 170)");
    drawLine(120, "rgba(170, 170, 170)");
    drawLine(210, "rgba(170, 170, 170)");

    svg.append("text")
        .text(d => "Sperm " + d.id)
        .attr('x', centerX)
        .attr('y', centerY + 125)
        .attr('text-anchor', 'middle');
}
