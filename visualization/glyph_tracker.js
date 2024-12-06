export function glyph_by_frame(espermatozoides) {
    d3.selectAll("g").remove();
    let container = document.querySelector('.glifos');
    let larguraTela = container.clientWidth;
    let altura = 600;
    // Configura a dimensão do SVG
    let svg = d3.select("svg")
        .attr('width', larguraTela*0.9)
        .attr('height', altura);

    svg.selectAll(".espermatozoide")
        .data(espermatozoides)
        .enter()
        .append("g")
        .attr("class", "espermatozoide");

    const espermatozoide = d3.selectAll(".espermatozoide");

    espermatozoide.each(function (d, i) {
        let g = d3.select(this);
        draw_route(g, d, i);
        for (var i = 0; i < d.frames.length; i++) {
            g.append("g")
                .attr("class", "glifo");

        }
        let glifo = d3.select(this).selectAll(".glifo");

        glifo.each(function (d, i) {

            rotate_glyph(d3.select(this), d.frames, i);
            draw_glyph(d3.select(this), d.frames[i]);
            tooltip_glyph(d3.select(this), d.frames[i]);
        });
    });


}
export function draw_route(path, d, i) {
    var pathData = "M" + d.route[0].x + "," + d.route[0].y;  // Começa o caminho no primeiro ponto

    // Para cada ponto seguinte, adiciona uma linha (L) para o próximo ponto
    for (var i = 1; i < d.route.length; i++) {
        pathData += " L" + d.route[i].x + "," + d.route[i].y;
    }
    // Adiciona o caminho no SVG
    path.append("path")
        .attr("class", "route")
        .attr("d", pathData)  // Define o caminho com a string gerada
        .attr("fill", "none")  // Não preenche o caminho (linha apenas)
        .attr("stroke", "blue")  // Cor da linha
        .attr("stroke-width", 2);
}
export function rotate_glyph(g, frames, i) {
    const centerX = frames[i].x;
    const centerY = frames[i].y;
    let angle = 0;

    if (i < frames.length - 1) {
        // Pega o próximo ponto (subsequente)
        const nextPoint = frames[i + 1];
        const nextX = nextPoint.x;
        const nextY = nextPoint.y;

        // Calculando o ângulo entre o ponto atual (d) e o próximo ponto (nextPoint)
        angle = (Math.atan2(nextY - centerY, nextX - centerX) * (180 / Math.PI)) + 90; // Convertendo de radianos para graus
    } else {
        const previousPoint = frames[i - 1];
        const previousX = previousPoint.x;
        const previousY = previousPoint.y;

        // Calculando o ângulo entre o ponto atual (d) e o anterior
        angle = (Math.atan2(centerY - previousY, centerX - previousX) * (180 / Math.PI)) + 90; // Convertendo de radianos para graus
    }
    g.attr("transform", `rotate(${angle}, ${centerX}, ${centerY})`);
}
export function draw_glyph(g, d) {
    const centerX = d.x;
    const centerY = d.y;

    const radii = {
        outer: 20,
        middle: 15,
        inner: 10,
    };

    // Fenda de 30 graus
    let startAngle = -5 * Math.PI / 6; // -30 graus
    let endAngle = 5 * Math.PI / 6;    // 30 graus

    // Função para criar semicírculos
    function drawSemiCircle(innerRadius, outerRadius, stroke, fill = "none", strokeWidth = 3, startAngle, endAngle) {
        g.append("path")
            .attr("d", d3.arc()
                .innerRadius(innerRadius)
                .outerRadius(outerRadius)
                .startAngle(startAngle)
                .endAngle(endAngle))
            .attr("fill", fill)
            .attr("stroke", stroke)
            .attr("stroke-width", strokeWidth)
            .attr("transform", d => `translate(${centerX}, ${centerY})`);;
    }
    let colorScale = d3.scaleSequential()
        .domain([0, 1]) // Ajuste o domínio conforme necessário
        .interpolator(d3.interpolateRainbow);

    // Desenhar semicírculos
    let color = colorScale(d.VSL);
    drawSemiCircle(radii.outer, radii.outer, "gray", "none", 0.5, startAngle, endAngle);
    drawSemiCircle(radii.middle, radii.middle, "black", "none", 0.5, startAngle, endAngle);
    drawSemiCircle(radii.inner, radii.inner, "gray", "rgba(200, 200, 200)", 0.5, startAngle, endAngle);
    drawSemiCircle(radii.inner - 2, radii.inner, "none", color, 0, startAngle, endAngle); // Semicírculo interno roxo
    drawSemiCircle(radii.outer, radii.outer + 2, "none", "rgba(200, 200, 200)", 0, startAngle, -1); // Semicírculo externo cinza
    drawSemiCircle(radii.inner - 2, 0, "none", "rgba(200, 200, 200)", 0, -Math.PI, Math.PI); // Semicírculo interno roxo
    drawSemiCircle(radii.inner - 2, 0, "gray", "white", 0.5, Math.PI / 6, -Math.PI / 6); //semicirculo branco
    const lineLength = 15;

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
        { x: centerX, y: centerY + 8 },
        { x: centerX - (lineLength / 6), y: centerY + 20 },
        { x: centerX + (lineLength / 6), y: centerY + 20 }
    ];
    drawTriangle(baseTriangle, "rgba(100, 100, 100, 0.5)", "rgba(100, 100, 100)", 0);

    const topTriangle = [
        { x: centerX, y: centerY - 24 },
        { x: centerX - (lineLength / 7), y: centerY - 20 },
        { x: centerX + (lineLength / 7), y: centerY - 20 }
    ];
    drawTriangle(topTriangle, "black", "black", 0);

    // Adicionar linhas em cruz
    const crossLength = 50;
    g.append("line") // Linha vertical
        .attr("x1", centerX)
        .attr("y1", centerY)
        .attr("x2", centerX)
        .attr("y2", centerY + 20)
        .attr("stroke", "white")
        .attr("stroke-width", 0.5);

    g.append("line") // Linha horizontal
        .attr("x1", centerX - 8)
        .attr("y1", centerY)
        .attr("x2", centerX + 8)
        .attr("y2", centerY)
        .attr("stroke", "white")
        .attr("stroke-width", 0.5);

    // Reta com círculos laranjas
    g.append("line")
        .attr("x1", centerX)
        .attr("y1", centerY + 8)
        .attr("x2", centerX)
        .attr("y2", centerY + 25)
        .attr("stroke", "black")
        .attr("stroke-width", 0.5);
    //.attr("transform", `rotate(35, ${centerX}, ${centerY + 8})`);

    // Círculos laranjas ao longo da linha
    const circlesCount = 3;
    const circleRadius = 1.2;

    for (let i = 1; i <= circlesCount; i++) {
        g.append("circle")
            .attr("cx", centerX)
            .attr("cy", centerY + 8 * i)
            .attr("r", circleRadius)
            .attr("stroke", "black")
            .attr("stroke-width", 0.5)
            .attr("fill", "orange");
        //.attr("transform", `rotate(35, ${centerX}, ${centerY + 8})`);
    }

    // Elipse no centro
    g.append("ellipse")
        .attr("cx", centerX)
        .attr("cy", centerY)
        .attr("rx", 3)
        .attr("ry", 5)
        .attr("fill", "lightgreen")
        .attr("stroke", "black")
        .attr("stroke-width", 0.1);
    //.attr("transform", `rotate(-25, ${centerX}, ${centerY})`);

    // Reta adicional rotacionada e transladada
    const length = 3;
    const radius = radii.outer; // Raio do círculo externo

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
            .attr("stroke-width", 1);
    }


    drawLine(0, "black");
    drawLine(180, "black");
    drawLine(150, "rgba(170, 170, 170)");
    drawLine(120, "rgba(170, 170, 170)");
    drawLine(210, "rgba(170, 170, 170)");

    /*glifo.append("text")
        .text("Sperm " + d.VSL)
        .attr('x', centerX)
        .attr('y', centerY + 25)
        .attr('text-anchor', 'middle');*/

}
export function tooltip_glyph(g, d) {
    // Cria um tooltip na body ou em outro contêiner
    var Tooltip = d3.select("body")  // Pode mudar para outro contêiner, se necessário
        .append("div")
        .style("opacity", 0)
        .attr("class", "tooltip")
        .style("background-color", "white") 
        .style("color", "black")
        .style("border-radius", "8px") 
        .style("padding", "8px 12px") 
        .style("font-size", "12px") 
        .style("box-shadow", "0 2px 6px rgba(0, 0, 0, 0.2)") 
        .style("position", "absolute")  // Necessário para o posicionamento absoluto
        .style("pointer-events", "none")  // Impede que o tooltip interfira com os eventos do mouse
        .style("transition", "opacity 0.2s ease-in-out");  // Suaviza a transição de visibilidade


    // Função para mostrar o tooltip quando o mouse passar sobre o item
    var mouseover = function () {
        Tooltip
            .style("opacity", 1);  // Torna o tooltip visível
    }

    // Função para mover o tooltip conforme o mouse se move
    var mousemove = function (event) {
        Tooltip
            .html("VCL: " + d.VCL + "<br>VSL: " + d.VSL + "<br>VAP: " + d.VAP + "<br>ALH: " + d.ALH + "<br>MAD: " + d.MAD)
            .style("left", (event.pageX + 20) + "px")  // Calcula a posição horizontal
            .style("top", (event.pageY ) + "px");  // Calcula a posição vertical
    }

    // Função para esconder o tooltip quando o mouse sai do item
    var mouseleave = function () {
        Tooltip
            .style("opacity", 0);  // Torna o tooltip invisível
    }

    // Adiciona os eventos de mouse sobre o glifo
    g.on("mouseover", mouseover)
        .on("mousemove", mousemove)
        .on("mouseleave", mouseleave);
}
