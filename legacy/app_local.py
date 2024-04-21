from flask import Flask, request, render_template_string, send_file, jsonify
import pandas as pd
import numpy as np
import io
import argparse

from ..utils.similarity import run_similarity

app = Flask(__name__)

@app.route('/')
def index():
    rows_with_indices = [{'index': index, 'data': row} for index, row in df.iterrows()]
    return render_template_string("""
    <html>
        <head>
            <title>Select Cells</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; position: relative; }
                #logo { position: absolute; top: 10px; right: 10px; }
                table { border-collapse: collapse; width: 100%; margin-top: 50px; }
                th, td { border: 1px solid black; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                button {
                    background-color: #4CAF50; color: white; padding: 10px; border: none; cursor: pointer;
                    width: auto; margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <img id="logo" src="https://serviall.cl/wp-content/uploads/2023/09/logo-serviall-1.webp" alt="logo-serviall" width=100>
            <h2>Select the most similar products</h2>
            <form action="/get-selected-values" method="post">
                <table>
                    <tr>
                        <th>Index</th> <!-- Adding header for index -->
                        {% for col_name in df.columns %}
                            <th>{{ col_name }}</th>
                        {% endfor %}
                    </tr>
                    {% for row in rows %}
                        <tr>
                            <td>{{ row.index }}</td> <!-- Display index without checkbox -->
                            {% for col_name, item in row.data.iteritems() %}
                                <td>
                                    <label>
                                        <input type="checkbox" name="selected_cells" value="{{ row.index }}_{{ col_name }}">
                                        {{ item }}
                                    </label>
                                </td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </table>
                <button type="submit">Download Selected Values as CSV</button>
            </form>
        </body>
    </html>
    """, rows=rows_with_indices, df=df)

@app.route('/get-selected-values', methods=['POST'])
def get_selected_values():
    selected_cells = request.form.getlist('selected_cells')
    selected_values = []

    # Extract the value from the DataFrame based on selected cells
    for cell in selected_cells:
        row_idx, col_name = cell.split('_')
        selected_values.append([row_idx, df.at[row_idx, col_name]])

    # Generate CSV
    if selected_values:
        selected_values = np.array(selected_values)
        bio = io.BytesIO()
        (
            pd.DataFrame(selected_values[:, 1], columns=['Selected Data'])
            .set_index(df.loc[selected_values[:, 0], :].index)
            .reset_index()
            .to_csv(bio, index=False)
        ) 
        bio.seek(0)
        return send_file(
            path_or_file=bio,
            mimetype='text/csv',
            as_attachment=True,
            download_name='selected_values.csv'
        )
    else:
        return jsonify(message="No cells selected")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, help='Path to the first dataset', default='datos empresa.xlsx')
    parser.add_argument('--path2', type=str, help='Path to the second dataset', default='datos mercado.xlsx')
    parser.add_argument('--output_file', type=str, help='Path to save the output file', default='final_df.csv')

    # Parse the arguments
    args = parser.parse_args()
    path_data_1 = args.path1
    path_data_2 = args.path2
    output_file = args.output_file

    # Run the similarity function
    run_similarity(path_data_1, path_data_2)

    # Load the DataFrame
    df = pd.read_csv(output_file, index_col=0)
    df.index = df.index.astype(str)
    
    # Run the Flask app
    app.run(debug=True)