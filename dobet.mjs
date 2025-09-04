import { exit } from "process"

const tok = process.argv[2]
const ch = process.argv[3]
const amt = process.argv[4]
const sd = process.argv[5]

//console.log(amt)

let res;

const acceptederrs = {
    "Please make at least 3 bets with current pair.": 0,
    "Please wait at least 20 seconds before the next randomize.": 1,
}

try {
    const fetchsend = fetch("https://duckdice.io/api/play?api_key=cad1d7be-8938-11f0-9a24-82ffecaea552", {
        "headers": {
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": "DuckDiceBot / 1.0.0",
            "cache-control": "no-cache",
            "host": "duckdice.io",
            "accept-encoding": "gzip, deflate, br",
            "connection": "keep-alive",
        },
        "body": JSON.stringify({
            "symbol": "ETC",
            "amount": amt,
            "chance": ch,
            "isHigh": tok == "above",
            "faucet": true
        }),
        "method": "POST"
    })
    //console.log('\n')
    res = await (await fetchsend).json();
    //console.log(res)
} catch (exc) {
    console.log(exc)
    exit(-1)
}

console.log(res.bet.number)

exit(+!(res.bet.result))