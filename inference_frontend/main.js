
const backend_enpoint = 'http://0.0.0.0:8000'

async function predict(image_path) 
{
    // figure out how to get BACKEND_PORT working for the fetch. it's in env
    // const response = await fetch(`${backend_enpoint}/predict?image_path=${image_path}`, 
    // {
    //     method: 'GET',
    //     body: JSON.stringify({
    //         image_path: image_path
    //     }),
    //     headers: 
    //     {
    //         'Content-Type': 'application/json'
    //     }
    // });

    const response = await fetch(`${backend_enpoint}/predict?image_path=${image_path}`);

    const data = await response.json();
    return data;
}
