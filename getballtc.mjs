import { exit } from "process"

let res

try {
    res = await (await fetch("https://duckdice.io/api/bot/user-info?api_key=cad1d7be-8938-11f0-9a24-82ffecaea552", {
        "headers": {
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": "DuckDiceBot / 1.0.0",
            "cache-control": "no-cache",
            "host": "duckdice.io",
            "accept-encoding": "gzip, deflate, br",
            "connection": "keep-alive",
        },
        "method": "GET"
    })).json();
} catch (exc) {
    console.log(exc)
    exit(-1)
}

//console.log(res)

const balses = res.balances

let balamnt = NaN

for (const bal of balses) {
    //console.log(bal)
    if (bal["currency"] === "ETC") {
        balamnt = bal["faucet"]
        break
    }
}

const balamntint = ((!isNaN(balamnt) ? 100000000 * balamnt >>> 0 : -1))

console.log(balamntint)