from dataclasses import replace
from time import perf_counter
from core.detector import Detection, predict
from core.reliability import quality
from core.sensor_simulator import proxy


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def merge(left,right,weights,threshold=.5):
    """Greedy one-to-one highest-IoU pairing. Penalize solitary evidence."""
    a,b=weights
    if not 0<threshold<=1 or min(weights)<0:
        raise ValueError("Invalid fusion settings")
    if min(weights)==0:
        return list(left if a else right if b else [])
    candidates=sorted(((-iou(d.box,e.box),i,j) for i,d in enumerate(left)
                       for j,e in enumerate(right) if d.cls==e.cls))
    used_left,used_right=set(),set()
    out=[]
    for neg_overlap,i,j in candidates:
        if -neg_overlap<threshold:
            break
        if i in used_left or j in used_right:
            continue
        d,e=left[i],right[j]
        w,v=a*d.score,b*e.score
        if w+v<=0:
            continue
        box=tuple((w*x+v*y)/(w+v) for x,y in zip(d.box,e.box))
        out.append(Detection(d.cls,(w+v)/(a+b),box,"matched RGB + proxy"))
        used_left.add(i)
        used_right.add(j)
    for items,used,weight in ((left,used_left,a),(right,used_right,b)):
        out.extend(replace(d,score=d.score*weight/(a+b),source="unmatched "+d.source)
                   for i,d in enumerate(items) if i not in used)
    return sorted(out,key=lambda d:-d.score)


def execute(model,rgb,mode="late_fusion",health=(1.,1.),size=640,confidence=.25,threshold=.5):
    start=perf_counter()
    if mode not in ("rgb_only","secondary_only","late_fusion","best","unavailable"):
        raise ValueError("Unknown mode")
    views={}
    for i,name in enumerate(("rgb","proxy")):
        if mode=="unavailable" or health[i]<=0 or (mode=="rgb_only" and i==1) or (mode=="secondary_only" and i==0):
            continue
        image=rgb if i==0 else proxy(rgb)
        if image.shape != rgb.shape:
            raise ValueError("Unaligned second view")
        weight=quality(image)*health[i]
        if weight>0:
            views[name]=(image,weight)
    if mode=="best" and views:
        name=max(views,key=lambda n:views[n][1])
        views={name:views[name]}
    predictions={name:predict(model,image,size,confidence,source=name)[0] for name,(image,_) in views.items()}
    weights={name:w for name,(_,w) in views.items()}
    if len(views)==2:
        boxes=merge(predictions["rgb"],predictions["proxy"],(weights["rgb"],weights["proxy"]),threshold)
    else:
        boxes=next(iter(predictions.values()),[])
    return [d for d in boxes if d.score>=confidence],dict(branches=list(views),weights=weights,
            seconds=perf_counter()-start,size=size,mode=mode if views else "unavailable")
