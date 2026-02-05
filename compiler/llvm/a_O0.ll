; ModuleID = 'a.cc'
source_filename = "a.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.Dog = type { i32, i8* }
%class.Animal = type { i32 (...)** }
%class.Cat = type { %class.Animal, i32, i8* }
%class.Lion = type { %class.Animal, i32, i8* }

$_ZN3CatC2EiPc = comdat any

$_ZN4LionC2EiPc = comdat any

$_ZN6AnimalC2Ev = comdat any

$_ZN3Cat5printEv = comdat any

$_ZN4Lion5printEv = comdat any

$_ZTV3Cat = comdat any

$_ZTS3Cat = comdat any

$_ZTS6Animal = comdat any

$_ZTI6Animal = comdat any

$_ZTI3Cat = comdat any

$_ZTV6Animal = comdat any

$_ZZN3Cat5printEvE3cnt = comdat any

$_ZTV4Lion = comdat any

$_ZTS4Lion = comdat any

$_ZTI4Lion = comdat any

@.str = private unnamed_addr constant [25 x i8] c"Dog age = %d, name = %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"dog\00", align 1
@__const._Z11PrintAnimalb.dog = private unnamed_addr constant %struct.Dog { i32 1, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i32 0, i32 0) }, align 8
@.str.2 = private unnamed_addr constant [4 x i8] c"cat\00", align 1
@.str.3 = private unnamed_addr constant [5 x i8] c"lion\00", align 1
@.str.4 = private unnamed_addr constant [44 x i8] c"\E8\AF\B7\E8\BE\93\E5\85\A5\E5\8A\A8\E7\89\A9\E7\B1\BB\E5\9E\8B\EF\BC\881: \E7\8B\97, 0: \E7\8C\AB\EF\BC\89: \00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.6 = private unnamed_addr constant [18 x i8] c"\E6\89\93\E5\8D\B0\E7\BB\93\E6\9E\9C: %d\0A\00", align 1
@_ZL10global_dog = internal global %struct.Dog { i32 2, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.9, i32 0, i32 0) }, align 8
@_ZTV3Cat = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI3Cat to i8*), i8* bitcast (void (%class.Cat*)* @_ZN3Cat5printEv to i8*)] }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
@_ZTS3Cat = linkonce_odr dso_local constant [5 x i8] c"3Cat\00", comdat, align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS6Animal = linkonce_odr dso_local constant [8 x i8] c"6Animal\00", comdat, align 1
@_ZTI6Animal = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_ZTS6Animal, i32 0, i32 0) }, comdat, align 8
@_ZTI3Cat = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZTS3Cat, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI6Animal to i8*) }, comdat, align 8
@_ZTV6Animal = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI6Animal to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)] }, comdat, align 8
@_ZZN3Cat5printEvE3cnt = linkonce_odr dso_local global i32 0, comdat, align 4
@.str.7 = private unnamed_addr constant [35 x i8] c"Cat age = %d, name = %s, cnt = %d\0A\00", align 1
@_ZTV4Lion = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI4Lion to i8*), i8* bitcast (void (%class.Lion*)* @_ZN4Lion5printEv to i8*)] }, comdat, align 8
@_ZTS4Lion = linkonce_odr dso_local constant [6 x i8] c"4Lion\00", comdat, align 1
@_ZTI4Lion = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @_ZTS4Lion, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI6Animal to i8*) }, comdat, align 8
@.str.8 = private unnamed_addr constant [26 x i8] c"Lion age = %d, name = %s\0A\00", align 1
@.str.9 = private unnamed_addr constant [7 x i8] c"global\00", align 1

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z8PrintDog3Dog(i32 %0, i8* %1) #0 {
  %3 = alloca %struct.Dog, align 8
  %4 = bitcast %struct.Dog* %3 to { i32, i8* }*
  %5 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %4, i32 0, i32 0
  store i32 %0, i32* %5, align 8
  %6 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %4, i32 0, i32 1
  store i8* %1, i8** %6, align 8
  %7 = getelementptr inbounds %struct.Dog, %struct.Dog* %3, i32 0, i32 0
  %8 = load i32, i32* %7, align 8
  %9 = getelementptr inbounds %struct.Dog, %struct.Dog* %3, i32 0, i32 1
  %10 = load i8*, i8** %9, align 8
  %11 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0), i32 noundef %8, i8* noundef %10)
  ret void
}

declare i32 @printf(i8* noundef, ...) #1

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z11PrintAnimalP6Animal(%class.Animal* noundef %0) #0 {
  %2 = alloca %class.Animal*, align 8
  store %class.Animal* %0, %class.Animal** %2, align 8
  %3 = load %class.Animal*, %class.Animal** %2, align 8
  %4 = bitcast %class.Animal* %3 to void (%class.Animal*)***
  %5 = load void (%class.Animal*)**, void (%class.Animal*)*** %4, align 8
  %6 = getelementptr inbounds void (%class.Animal*)*, void (%class.Animal*)** %5, i64 0
  %7 = load void (%class.Animal*)*, void (%class.Animal*)** %6, align 8
  call void %7(%class.Animal* noundef nonnull align 8 dereferenceable(8) %3)
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef i32 @_Z11PrintAnimalb(i1 noundef zeroext %0) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = alloca i8, align 1
  %3 = alloca i32, align 4
  %4 = alloca %struct.Dog, align 8
  %5 = alloca i32, align 4
  %6 = alloca %struct.Dog, align 8
  %7 = alloca %class.Cat*, align 8
  %8 = alloca i8*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %class.Lion*, align 8
  %11 = alloca i32, align 4
  %12 = zext i1 %0 to i8
  store i8 %12, i8* %2, align 1
  store i32 0, i32* %3, align 4
  %13 = load i8, i8* %2, align 1
  %14 = trunc i8 %13 to i1
  br i1 %14, label %15, label %32

15:                                               ; preds = %1
  store i32 2, i32* %3, align 4
  %16 = bitcast %struct.Dog* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %16, i8* align 8 bitcast (%struct.Dog* @__const._Z11PrintAnimalb.dog to i8*), i64 16, i1 false)
  store i32 0, i32* %5, align 4
  br label %17

17:                                               ; preds = %28, %15
  %18 = load i32, i32* %5, align 4
  %19 = icmp slt i32 %18, 3
  br i1 %19, label %20, label %31

20:                                               ; preds = %17
  %21 = bitcast %struct.Dog* %6 to i8*
  %22 = bitcast %struct.Dog* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %21, i8* align 8 %22, i64 16, i1 false)
  %23 = bitcast %struct.Dog* %6 to { i32, i8* }*
  %24 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %23, i32 0, i32 0
  %25 = load i32, i32* %24, align 8
  %26 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %23, i32 0, i32 1
  %27 = load i8*, i8** %26, align 8
  call void @_Z8PrintDog3Dog(i32 %25, i8* %27)
  br label %28

28:                                               ; preds = %20
  %29 = load i32, i32* %5, align 4
  %30 = add nsw i32 %29, 1
  store i32 %30, i32* %5, align 4
  br label %17, !llvm.loop !6

31:                                               ; preds = %17
  br label %69

32:                                               ; preds = %1
  store i32 1, i32* %3, align 4
  %33 = call noalias noundef nonnull i8* @_Znwm(i64 noundef 24) #7
  %34 = bitcast i8* %33 to %class.Cat*
  invoke void @_ZN3CatC2EiPc(%class.Cat* noundef nonnull align 8 dereferenceable(24) %34, i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0))
          to label %35 unwind label %50

35:                                               ; preds = %32
  store %class.Cat* %34, %class.Cat** %7, align 8
  %36 = call noalias noundef nonnull i8* @_Znwm(i64 noundef 24) #7
  %37 = bitcast i8* %36 to %class.Lion*
  invoke void @_ZN4LionC2EiPc(%class.Lion* noundef nonnull align 8 dereferenceable(24) %37, i32 noundef 1, i8* noundef getelementptr inbounds ([5 x i8], [5 x i8]* @.str.3, i64 0, i64 0))
          to label %38 unwind label %54

38:                                               ; preds = %35
  store %class.Lion* %37, %class.Lion** %10, align 8
  store i32 0, i32* %11, align 4
  br label %39

39:                                               ; preds = %47, %38
  %40 = load i32, i32* %11, align 4
  %41 = icmp slt i32 %40, 3
  br i1 %41, label %42, label %58

42:                                               ; preds = %39
  %43 = load %class.Cat*, %class.Cat** %7, align 8
  %44 = bitcast %class.Cat* %43 to %class.Animal*
  call void @_Z11PrintAnimalP6Animal(%class.Animal* noundef %44)
  %45 = load %class.Lion*, %class.Lion** %10, align 8
  %46 = bitcast %class.Lion* %45 to %class.Animal*
  call void @_Z11PrintAnimalP6Animal(%class.Animal* noundef %46)
  br label %47

47:                                               ; preds = %42
  %48 = load i32, i32* %11, align 4
  %49 = add nsw i32 %48, 1
  store i32 %49, i32* %11, align 4
  br label %39, !llvm.loop !8

50:                                               ; preds = %32
  %51 = landingpad { i8*, i32 }
          cleanup
  %52 = extractvalue { i8*, i32 } %51, 0
  store i8* %52, i8** %8, align 8
  %53 = extractvalue { i8*, i32 } %51, 1
  store i32 %53, i32* %9, align 4
  call void @_ZdlPv(i8* noundef %33) #8
  br label %71

54:                                               ; preds = %35
  %55 = landingpad { i8*, i32 }
          cleanup
  %56 = extractvalue { i8*, i32 } %55, 0
  store i8* %56, i8** %8, align 8
  %57 = extractvalue { i8*, i32 } %55, 1
  store i32 %57, i32* %9, align 4
  call void @_ZdlPv(i8* noundef %36) #8
  br label %71

58:                                               ; preds = %39
  %59 = load %class.Cat*, %class.Cat** %7, align 8
  %60 = icmp eq %class.Cat* %59, null
  br i1 %60, label %63, label %61

61:                                               ; preds = %58
  %62 = bitcast %class.Cat* %59 to i8*
  call void @_ZdlPv(i8* noundef %62) #8
  br label %63

63:                                               ; preds = %61, %58
  %64 = load %class.Lion*, %class.Lion** %10, align 8
  %65 = icmp eq %class.Lion* %64, null
  br i1 %65, label %68, label %66

66:                                               ; preds = %63
  %67 = bitcast %class.Lion* %64 to i8*
  call void @_ZdlPv(i8* noundef %67) #8
  br label %68

68:                                               ; preds = %66, %63
  br label %69

69:                                               ; preds = %68, %31
  %70 = load i32, i32* %3, align 4
  ret i32 %70

71:                                               ; preds = %54, %50
  %72 = load i8*, i8** %8, align 8
  %73 = load i32, i32* %9, align 4
  %74 = insertvalue { i8*, i32 } undef, i8* %72, 0
  %75 = insertvalue { i8*, i32 } %74, i32 %73, 1
  resume { i8*, i32 } %75
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull i8* @_Znwm(i64 noundef) #3

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN3CatC2EiPc(%class.Cat* noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1, i8* noundef %2) unnamed_addr #4 comdat align 2 {
  %4 = alloca %class.Cat*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i8*, align 8
  store %class.Cat* %0, %class.Cat** %4, align 8
  store i32 %1, i32* %5, align 4
  store i8* %2, i8** %6, align 8
  %7 = load %class.Cat*, %class.Cat** %4, align 8
  %8 = bitcast %class.Cat* %7 to %class.Animal*
  call void @_ZN6AnimalC2Ev(%class.Animal* noundef nonnull align 8 dereferenceable(8) %8) #9
  %9 = bitcast %class.Cat* %7 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV3Cat, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %9, align 8
  %10 = load i32, i32* %5, align 4
  %11 = getelementptr inbounds %class.Cat, %class.Cat* %7, i32 0, i32 1
  store i32 %10, i32* %11, align 8
  %12 = load i8*, i8** %6, align 8
  %13 = getelementptr inbounds %class.Cat, %class.Cat* %7, i32 0, i32 2
  store i8* %12, i8** %13, align 8
  ret void
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8* noundef) #5

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4LionC2EiPc(%class.Lion* noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1, i8* noundef %2) unnamed_addr #4 comdat align 2 {
  %4 = alloca %class.Lion*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i8*, align 8
  store %class.Lion* %0, %class.Lion** %4, align 8
  store i32 %1, i32* %5, align 4
  store i8* %2, i8** %6, align 8
  %7 = load %class.Lion*, %class.Lion** %4, align 8
  %8 = bitcast %class.Lion* %7 to %class.Animal*
  call void @_ZN6AnimalC2Ev(%class.Animal* noundef nonnull align 8 dereferenceable(8) %8) #9
  %9 = bitcast %class.Lion* %7 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Lion, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %9, align 8
  %10 = load i32, i32* %5, align 4
  %11 = getelementptr inbounds %class.Lion, %class.Lion* %7, i32 0, i32 1
  store i32 %10, i32* %11, align 8
  %12 = load i8*, i8** %6, align 8
  %13 = getelementptr inbounds %class.Lion, %class.Lion* %7, i32 0, i32 2
  store i8* %12, i8** %13, align 8
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main() #6 {
  %1 = alloca i32, align 4
  %2 = alloca i8, align 1
  %3 = alloca i32, align 4
  %4 = alloca %struct.Dog, align 8
  store i32 0, i32* %1, align 4
  %5 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([44 x i8], [44 x i8]* @.str.4, i64 0, i64 0))
  %6 = call i32 (i8*, ...) @__isoc99_scanf(i8* noundef getelementptr inbounds ([3 x i8], [3 x i8]* @.str.5, i64 0, i64 0), i8* noundef %2)
  %7 = load i8, i8* %2, align 1
  %8 = trunc i8 %7 to i1
  %9 = call noundef i32 @_Z11PrintAnimalb(i1 noundef zeroext %8)
  store i32 %9, i32* %3, align 4
  %10 = load i32, i32* %3, align 4
  %11 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([18 x i8], [18 x i8]* @.str.6, i64 0, i64 0), i32 noundef %10)
  %12 = bitcast %struct.Dog* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %12, i8* align 8 bitcast (%struct.Dog* @_ZL10global_dog to i8*), i64 16, i1 false)
  %13 = bitcast %struct.Dog* %4 to { i32, i8* }*
  %14 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %13, i32 0, i32 0
  %15 = load i32, i32* %14, align 8
  %16 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %13, i32 0, i32 1
  %17 = load i8*, i8** %16, align 8
  call void @_Z8PrintDog3Dog(i32 %15, i8* %17)
  ret i32 0
}

declare i32 @__isoc99_scanf(i8* noundef, ...) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN6AnimalC2Ev(%class.Animal* noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #4 comdat align 2 {
  %2 = alloca %class.Animal*, align 8
  store %class.Animal* %0, %class.Animal** %2, align 8
  %3 = load %class.Animal*, %class.Animal** %2, align 8
  %4 = bitcast %class.Animal* %3 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV6Animal, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN3Cat5printEv(%class.Cat* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca %class.Cat*, align 8
  store %class.Cat* %0, %class.Cat** %2, align 8
  %3 = load %class.Cat*, %class.Cat** %2, align 8
  %4 = load i32, i32* @_ZZN3Cat5printEvE3cnt, align 4
  %5 = add nsw i32 %4, 1
  store i32 %5, i32* @_ZZN3Cat5printEvE3cnt, align 4
  %6 = getelementptr inbounds %class.Cat, %class.Cat* %3, i32 0, i32 1
  %7 = load i32, i32* %6, align 8
  %8 = getelementptr inbounds %class.Cat, %class.Cat* %3, i32 0, i32 2
  %9 = load i8*, i8** %8, align 8
  %10 = load i32, i32* @_ZZN3Cat5printEvE3cnt, align 4
  %11 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([35 x i8], [35 x i8]* @.str.7, i64 0, i64 0), i32 noundef %7, i8* noundef %9, i32 noundef %10)
  ret void
}

declare void @__cxa_pure_virtual() unnamed_addr

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN4Lion5printEv(%class.Lion* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca %class.Lion*, align 8
  store %class.Lion* %0, %class.Lion** %2, align 8
  %3 = load %class.Lion*, %class.Lion** %2, align 8
  %4 = getelementptr inbounds %class.Lion, %class.Lion* %3, i32 0, i32 1
  %5 = load i32, i32* %4, align 8
  %6 = getelementptr inbounds %class.Lion, %class.Lion* %3, i32 0, i32 2
  %7 = load i8*, i8** %6, align 8
  %8 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([26 x i8], [26 x i8]* @.str.8, i64 0, i64 0), i32 noundef %5, i8* noundef %7)
  ret void
}

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { argmemonly nofree nounwind willreturn }
attributes #3 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nobuiltin nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { builtin allocsize(0) }
attributes #8 = { builtin nounwind }
attributes #9 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
