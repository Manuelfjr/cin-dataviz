export function glyph(esp_metrics) {

    d3.selectAll("#svg_glyph_metric").remove();

    let div_glifos = document.querySelector('.glifo_metric');
    let largura = div_glifos.clientWidth;
    let altura = div_glifos.clientHeight;

    let g = d3.selectAll(".glifo_metric").append("svg").attr("id", "svg_glyph_metric").attr("width", largura).attr("height", altura);

    let t = d3.select(".glifo_metric").select(".custom-header").html("Sumarização do espermatozóide #" + esp_metrics.tracker_id);
    console.log(t);
    const centerX = largura / 2;
    const centerY = 100;

    const radii = {
        outer: 60,
        middle: 45,
        inner: 30,
    };

    // Fenda de 30 graus
    let startAngle = -5 * Math.PI / 6; // -30 graus
    let endAngle = 5 * Math.PI / 6;    // 30 graus

    // Função para criar semicírculos
    function drawSemiCircle(innerRadius, outerRadius, stroke, fill = "none", strokeWidth = 9, startAngle, endAngle) {
        g.append("path")
            .attr("d", d3.arc()
                .innerRadius(innerRadius)
                .outerRadius(outerRadius)
                .startAngle(startAngle)
                .endAngle(endAngle))
            .attr("fill", fill)
            .attr("stroke", stroke)
            .attr("stroke-width", strokeWidth)
            .attr("transform", d => `translate(${centerX}, ${centerY})`);
    }

    let colorScale = d3.scaleSequential()
        .domain([0, 1])
        .interpolator(d3.interpolateRainbow);

    // Desenhar semicírculos
    let color = colorScale(esp_metrics.VSL);
    drawSemiCircle(radii.outer, radii.outer, "gray", "none", 1.5, startAngle, endAngle);
    drawSemiCircle(radii.middle, radii.middle, "black", "none", 1.5, startAngle, endAngle);
    drawSemiCircle(radii.inner, radii.inner, "gray", "rgba(200, 200, 200)", 1.5, startAngle, endAngle);
    drawSemiCircle(radii.inner - 6, radii.inner, "none", color, 0, startAngle, endAngle);
    drawSemiCircle(radii.outer, radii.outer + 6, "none", "rgba(200, 200, 200)", 0, startAngle, -1);
    drawSemiCircle(radii.inner - 6, 0, "none", "rgba(200, 200, 200)", 0, -Math.PI, Math.PI);

    //MAD
    let mad = esp_metrics.MAD * Math.PI / 270;
    drawSemiCircle(radii.inner - 6, 0, "gray", "white", 1.5, mad, -mad);

    const lineLength = 45;

    // Função para desenhar triângulos
    function drawTriangle(points, fill, stroke, strokeWidth) {
        g.append("polygon")
            .attr("points", points.map(p => `${p.x},${p.y}`).join(" "))
            .attr("fill", fill)
            .attr("stroke", stroke)
            .attr("stroke-width", strokeWidth);
    }

    // Triângulos da base
    const baseTriangle = [
        { x: centerX, y: centerY + 24 },
        { x: centerX - (lineLength / 6), y: centerY + 60 },
        { x: centerX + (lineLength / 6), y: centerY + 60 }
    ];
    drawTriangle(baseTriangle, "rgba(100, 100, 100, 0.5)", "rgba(100, 100, 100)", 0);

    const topTriangle = [
        { x: centerX, y: centerY - 72 },
        { x: centerX - (lineLength / 7), y: centerY - 60 },
        { x: centerX + (lineLength / 7), y: centerY - 60 }
    ];
    drawTriangle(topTriangle, "black", "black", 0);

    // Adicionar linhas em cruz
    const crossLength = 150;
    g.append("line") // Linha vertical
        .attr("x1", centerX)
        .attr("y1", centerY)
        .attr("x2", centerX)
        .attr("y2", centerY + 60)
        .attr("stroke", "white")
        .attr("stroke-width", 1.5);

    g.append("line") // Linha horizontal
        .attr("x1", centerX - 24)
        .attr("y1", centerY)
        .attr("x2", centerX + 24)
        .attr("y2", centerY)
        .attr("stroke", "white")
        .attr("stroke-width", 1.5);

    // Reta com círculos laranjas
    g.append("line")
        .attr("x1", centerX)
        .attr("y1", centerY + 24)
        .attr("x2", centerX)
        .attr("y2", centerY + 75)
        .attr("stroke", "black")
        .attr("stroke-width", 1.5);

    // Círculos laranjas ao longo da linha
    const circlesCount = 3;
    const circleRadius = 3.6;

    for (let i = 1; i <= circlesCount; i++) {
        g.append("circle")
            .attr("cx", centerX)
            .attr("cy", centerY + 24 * i)
            .attr("r", circleRadius)
            .attr("stroke", "black")
            .attr("stroke-width", 1.5)
            .attr("fill", "orange");
    }

    // Elipse no centro
    g.append("ellipse")
        .attr("cx", centerX)
        .attr("cy", centerY)
        .attr("rx", 9)
        .attr("ry", 15)
        .attr("fill", "lightgreen")
        .attr("stroke", "black")
        .attr("stroke-width", 0.3);

    // Reta adicional rotacionada e transladada
    const length = 9;
    const radius = radii.outer;

    function drawLine(angle, color) {
        let x1 = centerX + radius * Math.cos(angle * Math.PI / 180);
        let y1 = centerY + radius * Math.sin(angle * Math.PI / 180);
        let x2 = centerX + (radius + length) * Math.cos(angle * Math.PI / 180);
        let y2 = centerY + (radius + length) * Math.sin(angle * Math.PI / 180);

        g.append("line")
            .attr("x1", x1)
            .attr("y1", y1)
            .attr("x2", x2)
            .attr("y2", y2)
            .attr("stroke", color)
            .attr("stroke-width", 3);
    }

    drawLine(0, "black");
    drawLine(180, "black");
    drawLine(150, "rgba(170, 170, 170)");
    drawLine(120, "rgba(170, 170, 170)");
    drawLine(210, "rgba(170, 170, 170)");

    g.append("text")
        .attr("x", centerX)                    // Posição X do texto (horizontal)
        .attr("y", 200)                        // Posição Y inicial do texto (vertical)
        .attr("font-size", "16px")             // Tamanho da fonte
        .attr("fill", "black")                 // Cor do texto
        .attr("text-anchor", "middle")         // Alinha horizontalmente no centro
        .attr("dominant-baseline", "middle")   // Alinha verticalmente no centro
        .append("tspan")
        .attr("x", centerX)
        .attr("dy", "1.2em")                  // Desloca a primeira linha
        .text("VCL: " + esp_metrics.VCL)       // Primeira linha

        .append("tspan")
        .attr("x", centerX)
        .attr("dy", "1.2em")                  // Desloca a segunda linha
        .text("VSL: " + esp_metrics.VSL)      // Segunda linha

        .append("tspan")
        .attr("x", centerX)
        .attr("dy", "1.2em")                  // Desloca a terceira linha
        .text("VAP: " + esp_metrics.VAP)      // Terceira linha

        .append("tspan")
        .attr("x", centerX)
        .attr("dy", "1.2em")                  // Desloca a quarta linha
        .text("ALH: " + esp_metrics.ALH)      // Quarta linha

        .append("tspan")
        .attr("x", centerX)
        .attr("dy", "1.2em")                  // Desloca a quinta linha
        .text("MAD: " + esp_metrics.MAD);     // Quinta linha
}
