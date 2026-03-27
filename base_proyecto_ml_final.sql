drop table if exists desarrollo.base_rentabilidad;
create table desarrollo.base_rentabilidad as
select 
al.rut_comercio_liq AS id_comercio, 
al.codigo_sucursal_liq AS id_sucursal, 
al.fecha_tx_liq::date as fecha_trx,
al.fecha_devolucion::date AS fecha_devolucion,
al.marca_liq,
al.tipo_tx_liq AS tipo_tx,
fa.merchant_type_out AS mcc,
--CASE NACIONALIDAD--
case
    when al.marca_liq = 'AMEX' then
        case
            when al.emisor_liq in ('999999-BCO_INTERNACIONAL')
                 then 'Internacional'
            else 'Nacional'
        end
    else fa.nacionalidad_tx
end as nacionalidad,
al.categoria_producto,
fa.tarjeta_presente, 
dti.producto_id_marca AS tipo_tarjeta, -- TIPO DE TARJETA
fa.interchange_rate_out as ti_porcentual_BO1, 
count(al.codigo_mc_liq) as cantidad_tx,
sum(al.monto_total_venta_liq) as tpv, --VENTA TPV
sum(round(al.monto_comision_informa/1.19)) as merchant_neto, --MERCHANT NETO
sum(ti_porcentual_BO1*al.monto_total_venta_liq) as ti_monto_BO1, 
--Campos que construyen costos de marca-----------------------------
sum(
    coalesce(a.costo_autorizacion, 0)
  + coalesce(a.costo_autorizacion_dscto_visa, 0)
  + coalesce(a.costo_clearing, 0)
  + coalesce(a.costo_clearing_dscto_visa, 0)
  + coalesce(a.costo_trimestral, 0)
  + coalesce(a.costo_volumen_internacional, 0)
  + coalesce(a.costo_safety_net, 0)
  + coalesce(a.costo_non_authentication, 0)
  + coalesce(a.costo_network_contribution, 0)
  + coalesce(a.costo_administration_fee, 0)
  + coalesce(a.costo_conectivity_fee, 0)
  + coalesce(a.costo_addres_verification, 0)
  + coalesce(a.costo_dual_message, 0)
  + coalesce(a.costo_transaction_fee, 0)
) as costo_de_marca_final,
--Campos que construyen costos de marca-----------------------------
--ma_objetivo nunca menor que 1
sum(
  case
    when rc.ma_objetivo < 1 then 1
    else rc.ma_objetivo
  end
) as rentabilidad, -- RENTABILIDAD --------------------------------------------------
--CASE TASA INTERCAMBIO EN 0--
case
    /* ── NACIONAL ────────────────────────────────────────────── */
    when  ti_porcentual_BO1 = 0
      and merchant_type_out <> 9311
      and (
           /* AMEX nacional */
           (al.marca_liq = 'AMEX'
            and al.emisor_liq not in ('999999-BCO_INTERNACIONAL'))
           /* Otras marcas nacionales */
           or (al.marca_liq <> 'AMEX'
               and fa.nacionalidad_tx = 'Nacional')
          )
      and al.categoria_producto = 'CREDITO'         then 0.0114

    when  ti_porcentual_BO1 = 0
      and merchant_type_out <> 9311
      and (
           (al.marca_liq = 'AMEX'
            and al.emisor_liq not in ('999999-BCO_INTERNACIONAL'))
           or (al.marca_liq <> 'AMEX'
               and fa.nacionalidad_tx = 'Nacional')
          )
      and al.categoria_producto = 'DEBITO'          then 0.005

    when  ti_porcentual_BO1 = 0
      and merchant_type_out <> 9311
      and (
           (al.marca_liq = 'AMEX'
            and al.emisor_liq not in ('999999-BCO_INTERNACIONAL'))
           or (al.marca_liq <> 'AMEX'
               and fa.nacionalidad_tx = 'Nacional')
          )
      and al.categoria_producto = 'PREPAGO'         then 0.0094

    /* ── INTERNACIONAL ───────────────────────────────────────── */
    when  ti_porcentual_BO1 = 0
      and merchant_type_out <> 9311
      and (
           (al.marca_liq = 'AMEX'
            and al.emisor_liq in ('999999-BCO_INTERNACIONAL'))
           or (al.marca_liq <> 'AMEX'
               and fa.nacionalidad_tx = 'Internacional')
          )
      and al.marca_liq = 'AMEX'                     then 0.015

    when  ti_porcentual_BO1 = 0
      and merchant_type_out <> 9311
      and (
           (al.marca_liq = 'AMEX'
            and al.emisor_liq in ('999999-BCO_INTERNACIONAL'))
           or (al.marca_liq <> 'AMEX'
               and fa.nacionalidad_tx = 'Internacional')
          )
      and al.marca_liq = 'MASTERCARD'               then 0.02

    when  ti_porcentual_BO1 = 0
      and merchant_type_out <> 9311
      and (
           (al.marca_liq = 'AMEX'
            and al.emisor_liq in ('999999-BCO_INTERNACIONAL'))
           or (al.marca_liq <> 'AMEX'
               and fa.nacionalidad_tx = 'Internacional')
          )
      and al.marca_liq = 'VISA'                     then 0.0205

    when  ti_porcentual_BO1 = 0
      and merchant_type_out <> 9311
      and (
           (al.marca_liq = 'AMEX'
            and al.emisor_liq in ('999999-BCO_INTERNACIONAL'))
           or (al.marca_liq <> 'AMEX'
               and fa.nacionalidad_tx = 'Internacional')
          )
      and al.marca_liq = 'MAESTRO'                  then 0.02
    else ti_porcentual_BO1
end as ti_porcentual_BO2,
case 
	when (
        (al.marca_liq = 'AMEX'
         and al.emisor_liq not in ('999999-BCO_INTERNACIONAL'))
        or (al.marca_liq <> 'AMEX'
            and fa.nacionalidad_tx = 'Nacional')
     )
     and ti_porcentual_BO2 > 0.0114
then 0.0114
	else ti_porcentual_BO2
end as ti_porcentual_BO,
case
    /* NACIONAL (mismo bloque de condiciones que en BO2) */
    when  ti_porcentual_BO1 = 0
      and merchant_type_out <> 9311
      and (
           (al.marca_liq = 'AMEX'
            and al.emisor_liq not in ('999999-BCO_INTERNACIONAL'))
           or (al.marca_liq <> 'AMEX'
               and fa.nacionalidad_tx = 'Nacional')
          )
      then (ti_porcentual_BO * tpv)

    /* INTERNACIONAL para cada marca */
    when  ti_porcentual_BO1 = 0
      and merchant_type_out <> 9311
      and (
           (al.marca_liq = 'AMEX'
            and al.emisor_liq in ('999999-BCO_INTERNACIONAL'))
           or (al.marca_liq <> 'AMEX'
               and fa.nacionalidad_tx = 'Internacional')
          )
      then (ti_porcentual_BO * tpv)

    /* Tope de 1,14 % para las nacionales */
    when (
          (al.marca_liq = 'AMEX'
           and al.emisor_liq not in ('999999-BCO_INTERNACIONAL'))
          or (al.marca_liq <> 'AMEX'
              and fa.nacionalidad_tx = 'Nacional')
         )
         and ti_porcentual_BO2 > 0.0114
      then (ti_porcentual_BO * tpv)
    else (ti_monto_BO1)
end as tasa_intercambio_final,
round(merchant_neto - tasa_intercambio_final - costo_de_marca_final) as margen_adquirente
from datamarts.adq_liquidaciones al
left join datamarts.fact_adquirencia fa on (al.codigo_mc_liq=fa.tx_codigo_mc)
left join sof.reportes_comercios rc on (al.id=rc.id) -- SOLICITAR TABLA MAX MIN 
left join rentabilidad.costos_de_marcas_dolar_fijo a on (al.id=a.id)
left join datamarts.dim_tasa_intercambio dti on (fa.llave=dti.llave)
where 1=1
and al.tipo_tx_liq not in ('REVERSA_VENTA CREDITO',	'ANULACION_PARCIAL DEBITO',	'ANULACION_RECARGA_TEL DEBITO',	'ANULACION_LUZ CREDITO',	
'REVERSA_ANULACION DEBITO',	'CONTRACARGO_CHECKOUT DEBITO',	'CONTRACARGO_CHECKOUT CREDITO',	'ANULACION_AGUA DEBITO',	'ANULACION_RECARGA_TEL CREDITO',	
'ANULACION_FONASA CREDITO',	'ANULACION DEBITO',	'REVERSA_VENTA PREPAGO',	'ANULACION',	'ANULACION_FONASA DEBITO',	'CONTRACARGO_VENTA CREDITO',	
'ANULACION_PARCIAL',	'DEMORADO CREDITO',	'ANULACION PREPAGO',	'ANULACION_PARCIAL CREDITO',	'ANULACION_PARCIAL PREPAGO',	'REVERSA_ANULACION CREDITO',	
'ANULACION_LUZ DEBITO',	'CONTRACARGO_CHECKOUT PREPAGO',	'ANULACION_AGUA CREDITO',	'REVERSA_VENTA DEBITO',	'ANULACION CREDITO',	'CONTRACARGO_VENTA DEBITO',	
'REVERSA_VENTA',	'REVERSA_ANULACION PREPAGO',	'CONTRACARGO_VENTA PREPAGO', 'REVERSA_CHECKOUT CREDITO')
and al.fecha_devolucion >= '2025-01-01'::timestamp
and al.fecha_devolucion < '2026-01-01'::timestamp 
and al.rut_comercio_liq in ( select distinct sc.rut_comercio
from sof.segmentacion_clientes sc 
where clasificacion_final = 'Mercado PSP y Marketplace' 
and origen_segmentacion != 'COPEC' -- copec no es psp
)
group by al.rut_comercio_liq,
    al.codigo_sucursal_liq,
    al.fecha_tx_liq::date,
    al.fecha_devolucion::date,
    al.marca_liq,
    al.tipo_tx_liq,
    fa.merchant_type_out,
    al.emisor_liq,
    fa.nacionalidad_tx,
    al.categoria_producto,
    fa.tarjeta_presente,
    dti.producto_id_marca,
    fa.interchange_rate_out;
