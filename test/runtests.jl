using DunklZernikeExpansions
using LinearAlgebra

@assert map(DunklZernikeExpansions.polyDim, 0:5) == [1, 3, 6, 10, 15, 21]

maxdeg = 10
index_triplets = [(m,(deg-m)÷2,even) for deg=0:maxdeg for m=deg:-2:0 for even in (true,false) if (m>0 || even)]
indices = map(v -> DunklZernikeExpansions.pairing(v...), index_triplets)
@assert indices==1:DunklZernikeExpansions.polyDim(maxdeg)
@assert index_triplets == map(DunklZernikeExpansions.inversepairing, indices)

d = 4
v = randn(DunklZernikeExpansions.polyDim(d))
f1 = DZFun((1,2,3),d,v)
f2 = DZFun((1,2,3.0),v)
@assert f1 == f2
f3 = DZFun([1,2,3],v)
f4 = DZFun([1.0,2.0,3.0],[v;0.0;0.0;0.0;0.0])
f5 = DZFun([1.0,2.0,3.0],d,v)
@assert f3 == f4
@assert f3 == f5

g1 = DZPoly(DZParam(-1/2,-1/3,0),2,1,true)
g2 = DZPoly((-1/2,-1/3,0),2,1,true)
g3 = DZPoly([-1/2,-1/3,0],2,1,true)
@assert g1==g2
@assert g1==g3

points = [randn(2) for i=1:100]
parameters = [DZParam(10*rand(3)-[1.0,1.0,1.0]...) for i = 1:100]

# Test raise via evalDZ
d = 20
v = randn(DunklZernikeExpansions.polyDim(d))

for param in parameters 
	f = DZFun(param,d,v)
	rf = DunklZernikeExpansions.raise(f)
	for point in points
		@assert evalDZ(f,point[1],point[2]) ≈ evalDZ(rf,point[1],point[2])
	end
end

# Test lower via evalDZ
d = 20
v = randn(DunklZernikeExpansions.polyDim(d))
for param in parameters 
	shiftedparam = DZParam(param.γ1,param.γ2,param.α+1.0)
	f = DZFun(shiftedparam,d,v)
	lf = DunklZernikeExpansions.lower(f)
	for point in points
		@assert evalDZ(f,point[1],point[2]) ≈ evalDZ(lf,point[1],point[2])
	end
end

# Test whether lower and raise are the inverse of one another
for param in parameters
	f = DZFun(param,d,v)
	r = DunklZernikeExpansions.lower(DunklZernikeExpansions.raise(f))-f
	@assert norm(r.coefficients) < 1e-10
end
for param in parameters
	shiftedparam = DZParam(param.γ1,param.γ2,param.α+1.0)
	f = DZFun(shiftedparam,d,v)
	r = DunklZernikeExpansions.raise(DunklZernikeExpansions.lower(f))-f
	@assert norm(r.coefficients) < 1e-10
end

# Test DunklX and DunklY via raise
# Test commutativity
for param in parameters
	f = DZFun(param,d,v)
	
	DxR = DunklZernikeExpansions.DunklX(DunklZernikeExpansions.raise(f))
	RDx = DunklZernikeExpansions.raise(DunklZernikeExpansions.DunklX(f))
	
	DyR = DunklZernikeExpansions.DunklY(DunklZernikeExpansions.raise(f))
	RDy = DunklZernikeExpansions.raise(DunklZernikeExpansions.DunklY(f))

	for point in points
		@assert evalDZ(DxR,point[1],point[2]) ≈ evalDZ(RDx,point[1],point[2])
		@assert evalDZ(DyR,point[1],point[2]) ≈ evalDZ(RDy,point[1],point[2])
		@assert evalDZ(DunklZernikeExpansions.DunklX(DunklZernikeExpansions.DunklY(f)),point[1],point[2]) ≈ evalDZ(DunklZernikeExpansions.DunklY(DunklZernikeExpansions.DunklX(f)),point[1],point[2])
	end
end

# Test mbx1 via evalDZ
for param in parameters
	f = DZFun(param,d,v)
	fbx1 = mbx1(f)
	fbx2 = mbx2(f)

	for point in points
		@assert evalDZ(fbx1,point[1],point[2]) ≈ point[1]*evalDZ(f,point[1],point[2])
		@assert evalDZ(fbx2,point[1],point[2]) ≈ point[2]*evalDZ(f,point[1],point[2])
	end
end

# Test sym and skew
for param in parameters
	f = DZFun(param,d,v)
	for point in points
		val = evalDZ(f,point[1],point[2])
		valsigma1star = evalDZ(f,-point[1],point[2])
		valsigma2star = evalDZ(f,point[1],-point[2])
		@assert evalDZ(sym(f,1),point[1],point[2]) ≈ (val+valsigma1star)/2.0
		@assert evalDZ(skew(f,1),point[1],point[2]) ≈ (val-valsigma1star)/2.0
		@assert evalDZ(sym(f,2),point[1],point[2]) ≈ (val+valsigma2star)/2.0
		@assert evalDZ(skew(f,2),point[1],point[2]) ≈ (val-valsigma2star)/2.0
	end
end
