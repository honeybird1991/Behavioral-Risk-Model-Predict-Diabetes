$(function () {
    $('form').on('submit', function (event) {
    // using this page stop being refreshing 
    event.preventDefault();
      $.ajax(
        {
            type: "GET",
            dataType: "json",
            async: false,
            url: "http://0.0.0.0:5000/",//url
            data: $('form').serialize(),
            success: function(result) {
                if (!result.success) {
                  document.getElementById("result").innerHTML=result.reason;
                }
            }
        }
        );
    });
})