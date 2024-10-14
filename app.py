from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    variaveis = {
        'centerX': 300,
        'centerY': 300,
        'radii': {
            'outer': 200,
            'middle': 150,
            'inner': 100,
        },
        'lineLength': 150,
        'circlesCount': 3,
        'circleRadius': 12,
    }
    return render_template('glyph.html', **variaveis)

if __name__ == '__main__':
    app.run(debug=True)
