import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import os
import base64
from urllib.parse import quote as urlquote
import ProfitOptimizer as optim


# -------------------------- UI Helpers ---------------------------- #

def build_banner():
    return html.Div(
        id='banner',
        className='banner',
        children=[
            html.Img(src=app.get_asset_url('dash_logo.png')),
        ],
    )



# -------------------------- LOAD DATA ---------------------------- #

UPLOAD_DIRECTORY = "BusinessFiles"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def save_file(name, content):
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        print("uploading files in: ", os.path.join(UPLOAD_DIRECTORY, name))
        fp.write(base64.decodebytes(data))


def uploaded_files():
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        print("uploaded filepath: ",path)
        if os.path.isfile(path):
            files.append(filename)
    return files

def file_download_link(filename):
    location = "download/{}".format(urlquote(filename))
    return html.A("Original Retail data: "+filename, href=location)

# -------------------------- PROJECT DASHBOARD ---------------------------- #

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder='assets')

@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        n = len(files) - 1
        filename = files[n]
        csv_files_path = os.path.join(UPLOAD_DIRECTORY, filename)
        final_price_list, profit_wo_price_change_effect, profit_with_price_change_effect, change_in_total_price, perc_lift_in_profit, perc_change_in_price, change_in_total_units_sold, perc_change_in_sales=optim.findOptimalPrice(csv_files_path)
        dash_text1 = '''Profit without Optimization: '''+"{:.2f}".format(profit_wo_price_change_effect)
        dash_text2 = '''Net Estimated Profit: '''+"{:.2f}".format(profit_with_price_change_effect)+ ''' , LIFT by ['''+"{:.2f}".format(perc_lift_in_profit)+''' %]'''
        dash_text3 = '''Increase in Total Pricing: '''+"{:.2f}".format(change_in_total_price)+ ''', ['''+"{:.2f}".format(perc_change_in_price)+''' %]'''
        dash_text4 = '''Increase in Total Units Sold: '''+"{:.2f}".format(change_in_total_units_sold)+ ''', ['''+"{:.2f}".format(perc_change_in_sales)+''' %]'''

        final_price_list.round(2)
        return [
        html.Li(file_download_link(filename)),
        html.P(id='net_profit',children=dash_text1),
        html.P(id='original_profit',children=dash_text2),
        html.P(id='price_change',children=dash_text3),
        html.P(id='volume_change', children=dash_text4),
        dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in final_price_list.columns],
        data=final_price_list.to_dict('records'),
        fixed_rows={'headers': True},
        # page_size=10,  # we have less data in this example, so setting to 20
        style_table={'height': '300px', 'width': '1000px', 'overflowY': 'auto'},
        style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 95}
    )]
server = app.server

app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[
    html.H1(
        children=[
            build_banner()
        ]
    ),
    html.H2("Upload Retail Data"),
    dcc.Upload(
        id="upload-data",
        children=html.Div(
            ["Drag and drop or click to select a file to upload."]
        ),
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
        },
        multiple=True,
    ),
    html.H2("Optimal Price & Profit"),
    html.Ul(id="file-list")

])

# -------------------------- MAIN ---------------------------- #


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8088, debug=True, use_reloader=True)
