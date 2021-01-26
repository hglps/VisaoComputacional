import PySimpleGUI as sg
import cv2
import numpy as np
import disparity as disp

def main():
    sg.theme("LightGreen")

    layout = [
        [sg.Text("3D Reconstruction", size=(60, 1), justification="center")],
        [sg.Image(filename="", key="-IMAGE-")],
        [
            sg.Text('Left image:'), sg.Input(key='_leftpath_'), sg.FileBrowse()
        ],
        [
            sg.Text('Right image:'), sg.Input(key='_rightpath_'), sg.FileBrowse()
        ],
        [
            sg.Text("Max disparity", size=(10, 1), key="-maxDisp-"),
            sg.Slider(
                (1, 10),
                5,
                1,
                orientation="h",
                size=(40, 15),
                key="-maxDisp SLIDER-",
            ),
        ],
        [
            sg.Text("Block size + 1", size=(10, 1), key="-blockSize-"),
            sg.Slider(
                (3, 11),
                3,
                2,
                orientation="h",
                size=(40, 15),
                key="-blockSize SLIDER-",
            ),
            sg.Text("Uniqueness Ratio", size=(20, 1), key="-uniquenessRatio-"),
            sg.Slider(
                (5, 15),
                5,
                1,
                orientation="h",
                size=(40, 15),
                key="-uniquenessRatio SLIDER-",
            ),
        ],
        [
            sg.Text("Speckle window size", size=(20, 1), key="-speckleWindowSize-"),
            sg.Slider(
                (50, 200),
                50,
                1,
                orientation="h",
                size=(40, 15),
                key="-speckleWindowSize SLIDER-",
            ),
            sg.Text("Speckle Range", size=(20, 1), key="-speckleRange-"),
            sg.Slider(
                (1, 2),
                1,
                1,
                orientation="h",
                size=(40, 15),
                key="-speckleRange SLIDER-",
            ),
        ],
        [sg.Button('Get disparity map', size=(20, 1))],
        [
            sg.Text("Threshold", size=(10, 1), key="-threshold-"),
            sg.Slider(
                (50, 800),
                200,
                1,
                orientation="h",
                size=(40, 15),
                key="-threshold SLIDER-",
            ),
        ],
        [sg.Button('Get 3D Body Reconstruction', size=(20, 1)), sg.Button('Get Full 3D Reconstruction', size=(20,1))],
        [sg.Button("Exit", size=(20, 1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("3D Reconstruction", layout, location=(800, 500))

    default_img = np.zeros((360, 240))

    disparityFlag = False

    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == 'Get disparity map':
            print("Started calculating the disparity map")
            if values['_leftpath_'] != '' and values['_rightpath_'] != '':
                disparityFlag = True

                disparity_map, output_points, output_colors = disp.gen_disparity_map(values['_leftpath_'], values['_rightpath_'],
                values['-maxDisp SLIDER-'], values['-blockSize SLIDER-'] - 1, values['-uniquenessRatio SLIDER-'], values['-speckleWindowSize SLIDER-'],
                values['-speckleRange SLIDER-'])
            else: sg.Popup('Image path missing! Try filling it out!')
        
        if event == 'Get 3D Body Reconstruction':
            if disparityFlag == True:
                status, b_points, b_colors = disp.reconstruct_3d('threshold',output_points, output_colors,
                values['-threshold SLIDER-'])
                sg.Popup('3D Body Reconstruction is done.')
            else: sg.Popup('You must generate a disparity map first.')

        if event == 'Get Full 3D Reconstruction':
            if disparityFlag == True:
                status, b_points, b_colors = disp.reconstruct_3d('full',output_points, output_colors,
                values['-threshold SLIDER-'])
                sg.Popup('Full 3D Reconstruction is done.')
            else: sg.Popup('You must generate a disparity map first.')


        if disparityFlag == True:
            disparity_map = cv2.resize(disparity_map, (240, 360), interpolation=cv2.INTER_CUBIC)
            imgbytes = cv2.imencode(".png", disparity_map)[1].tobytes()
        else: 
            imgbytes = cv2.imencode(".png", default_img)[1].tobytes()

        window["-IMAGE-"].update(data=imgbytes)

    window.close()

main()