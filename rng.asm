
.code
getrngterry PROC
rdtsc
mov eax, eax
shl rdx, 32
add rax, rdx
ret
getrngterry ENDP

END