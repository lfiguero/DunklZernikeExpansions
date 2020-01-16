using DunklZernikeExpansions
using LinearAlgebra

param = [-8/9,-7/8,-0.5]
coeflist = [randn(DunklZernikeExpansions.polyDim(deg)) for deg=0:40 for j=1:1000]
relationarray = zeros(length(coeflist))
for i=1:length(relationarray)
	f = DZFun(param,coeflist[i])
	fproj = DunklZernikeExpansions.project(f,0)
	fdx1 = Dunklx1(f)
	fdx2 = Dunklx2(f)

	usualnorm = sqrt(DunklZernikeExpansions.DZFunInner(f,f) + DunklZernikeExpansions.DZFunInner(fdx1,fdx1) + DunklZernikeExpansions.DZFunInner(fdx2,fdx2))
	projnorm = sqrt(DunklZernikeExpansions.DZFunInner(fproj,fproj) + DunklZernikeExpansions.DZFunInner(fdx1,fdx1) + DunklZernikeExpansions.DZFunInner(fdx2,fdx2))
	relationarray[i] = usualnorm/projnorm
end
println(minimum(relationarray))
println(maximum(relationarray))
