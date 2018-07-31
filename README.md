## Style It!

Style images with convolutional neural nets with *online* parameters tuning.
"Static" realization based on Stanford cs231n/assignment3/ StyleTransfer with Tensorflow notebook.

Run with `python main.py` from terminal.
Main command line arguments:

    --default-style-decay - default value for style importance.
    --default-content-weight - default value for content importance.
    --content-max-side - max side of content image.
    --style-max-side  - max side of style image.
    --save-each - step size for saving progress.
    --update-each - step size for updating result.

#### Usage example:
<img src='samples/usage.gif' width=600px>

#### Generated samples:
<table>
    <tr>
        <th>Content</th>
        <th>Style</th>
        <th>Progress</th>
    </tr>
    <tr>
        <td> <img src='contents/lion.jpg' width=300px></td>
        <td> <img src='styles/style3.jpg' width=300px></td>
        <td>
            <img src='samples/lion.gif' width=250px>  
            <img src='samples/lion2.gif' width=250px>  
        </td>
    </tr>
    <tr>
        <td> <img src='contents/lena.jpg' width=300px></td>
        <td> <img src='styles/style2.jpg' width=300px></td>
        <td> <img src='samples/lena_abstract.gif' width=250px>  </td>
    </tr>
    <tr>
        <td> <img src='contents/lena.jpg' width=300px></td>
        <td> <img src='styles/style3.jpg' width=300px></td>
        <td> <img src='samples/lena.gif' width=250px>  </td>
    </tr>
</table>
