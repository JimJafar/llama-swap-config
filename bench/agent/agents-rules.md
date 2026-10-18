# Working in long tasks

- Put `timeout` at the start of every command that could take more than a few seconds (`timeout 120 <command>` unless there is a reason for longer). A timeout covers only the command it starts, so wrap each slow part of a pipeline or `;` chain. Never search `/` or mounted drives. Size your own tests and checks to finish in under a minute.
- If the same fix-and-retest loop has not converged after about three rounds, step back: question your assumptions, change the approach, or report what is left instead of trying another variation.
- To check how something looks, look at it: take a screenshot and open it with `read` if you can see images. Don't compute pixel statistics or invent other proxy metrics for qualities a person judges. Check once, fix what is clearly wrong, then hand back to the user.
- Keep command output short (`tail`, `head`, `grep`, a summary): everything you print is read again on every later step.
